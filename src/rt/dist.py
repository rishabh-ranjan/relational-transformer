"""Distributed-setup helpers shared by the train and eval drivers."""

import fnmatch
import os
import socket


def disable_gdr_on_ampere():
    """Turn off GPUDirect RDMA on the `ampere*` nodes, where it hangs.

    A multi-node job on those nodes wedges in the first inter-node collective
    (the init-time model broadcast): every rank enqueues it, none ever starts
    it, and the job burns its whole NCCL watchdog timeout before dying. Must be
    called before ``init_process_group``.
    """
    if fnmatch.fnmatch(socket.getfqdn(), "ampere*.stanford.edu"):
        os.environ["NCCL_NET_GDR_LEVEL"] = "0"
