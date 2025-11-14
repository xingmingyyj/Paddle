# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod

import paddle
import paddle.distributed as dist
from paddle.distributed.fleet.utils.log_util import logger


class AbstractCommunicator(ABC):
    @abstractmethod
    def communicate(self, comm_tasks, state, context):
        pass


def get_target_tensor(target_state_dict, read_item):
    use_dist = True if paddle.distributed.get_world_size() > 1 else False
    if any(isinstance(k, tuple) for k in target_state_dict):
        key = (read_item.tensor_name, read_item.dst_global_offset)
    else:
        key = read_item.tensor_name
    target_tensor = (
        target_state_dict[key]._local_value()
        if use_dist and target_state_dict[key].is_dist()
        else target_state_dict[key]
    )
    return target_tensor


def slice_tensor(tensor, slice_begin, slice_shape):
    # If slice_shape is empty, the tensor is 0-dimensional (scalar); return it as is.
    if len(slice_shape) == 0:
        assert len(tensor.shape) == 0, (
            "Only 0-dimensional tensor supports empty slice_shape."
        )
        return tensor
    slice_end = [
        start + length for start, length in zip(slice_begin, slice_shape)
    ]
    axes = list(range(tensor.ndim))
    return paddle.slice(tensor, axes=axes, starts=slice_begin, ends=slice_end)


class SendRecvCommunicator(AbstractCommunicator):
    def communicate(self, comm_tasks, state, context):
        cur_rank = context['rank']
        process_group = context['process_group']
        use_group = context['use_group']

        source_state_dict = state['source_state_dict']
        target_state_dict = state['target_state_dict']

        source_tensor_slices = {}
        target_tensor_slices = {}
        local_copy_task = set()
        for tensor_name, read_items in comm_tasks.items():
            need_clear = set()
            for item in read_items:
                if cur_rank == item.src_rank:
                    src_tensor = source_state_dict[item.file_name][
                        item.tensor_name
                    ]
                    src_chunk_tensor = slice_tensor(
                        src_tensor, item.src_local_offset, item.slice_shape
                    ).clone()
                    source_tensor_slices[item] = src_chunk_tensor.contiguous()
                    need_clear.add(src_tensor)
                if cur_rank in item.dst_rank:
                    if cur_rank == item.src_rank:
                        local_copy_task.add(item)
                        target_tensor_slices[item] = source_tensor_slices[item]
                    else:
                        dst_chunk_tensor = paddle.zeros(
                            item.slice_shape, dtype=item.dtype
                        )
                        target_tensor_slices[item] = dst_chunk_tensor
            for tensor in need_clear:
                tensor._clear_to_zero_allocation()

        send_recv_ops = []
        for tensor_name, read_items in comm_tasks.items():
            for item in read_items:
                if item.src_rank == cur_rank:
                    for rank in item.dst_rank:
                        if rank == cur_rank:
                            continue
                        send_t = source_tensor_slices[item]
                        if use_group:
                            send_op = dist.P2POp(dist.isend, send_t, rank)
                            send_recv_ops.append(send_op)
                        else:
                            dist.send(send_t, rank)
                if cur_rank in item.dst_rank:
                    if item.src_rank == cur_rank:
                        continue
                    recv_t = target_tensor_slices[item]
                    if use_group:
                        recv_op = dist.P2POp(dist.irecv, recv_t, item.src_rank)
                        send_recv_ops.append(recv_op)
                    else:
                        dist.recv(recv_t, item.src_rank)

        if use_group:
            logger.info("Starting to send/recv tensors using P2POp.")
            task_handles = dist.batch_isend_irecv(send_recv_ops)
            for task in task_handles:
                task.wait()
            logger.info("Send/Recv tensors finished.")

        for item in source_tensor_slices:
            if item not in local_copy_task:
                source_tensor_slices[item]._clear()

        del source_tensor_slices

        for item in target_tensor_slices:
            dst_tensor = get_target_tensor(target_state_dict, item)
            if not dst_tensor._is_initialized():
                buffer = paddle.zeros_like(dst_tensor)
                buffer._share_buffer_to(dst_tensor)
            dst_chunk_tensor = slice_tensor(
                dst_tensor, item.dst_local_offset, item.slice_shape
            )
            cur_chunk_tensor = target_tensor_slices[item]
            if dst_chunk_tensor.place != cur_chunk_tensor.place:
                cur_chunk_tensor = cur_chunk_tensor.to(dst_chunk_tensor.place)
            paddle.assign(cur_chunk_tensor, dst_chunk_tensor)

        paddle.distributed.barrier(process_group)
        logger.info("All communication tasks completed.")
