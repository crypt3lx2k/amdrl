#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import configs.distributed

import subprocess
import time

def main (FLAGS, CHILD_FLAGS):
    cluster = configs.distributed.get_cluster(FLAGS)
    processes = {'ps' : [], 'worker' : []}

    # Launch ps tasks
    for task_index, parameter_task in enumerate(cluster['ps']):
        print(task_index, parameter_task)

        parameter_server = subprocess.Popen (
            [
                'python', 'parameter_server.py',
                '--task_index',   '{}'.format(task_index),
                '--ps_tasks',     '{}'.format(FLAGS.ps_tasks),
                '--worker_tasks', '{}'.format(FLAGS.worker_tasks),
            ] + CHILD_FLAGS
        )

        processes['ps'].append(parameter_server)
        time.sleep(FLAGS.launch_delay)

    # Launch worker tasks
    for task_index, worker_task in enumerate(cluster['worker']):
        print(task_index, worker_task)

        worker = subprocess.Popen (
            [
                'python', 'worker.py',
                '--task_index',   '{}'.format(task_index),
                '--ps_tasks',     '{}'.format(FLAGS.ps_tasks),
                '--worker_tasks', '{}'.format(FLAGS.worker_tasks),
            ] + CHILD_FLAGS
        )

        processes['worker'].append(worker)
        time.sleep(FLAGS.launch_delay)

    try:
        while True:
            # Poll worker processes
            live_workers = filter(lambda p : p.poll() is None, processes['worker'])

            # Quit if all workers are finished
            if len(live_workers) == 0:
                break

            time.sleep(FLAGS.poll_delay)

        print('All workers finished, cleaning up and exiting.')
    except KeyboardInterrupt as e:
        # Catch CTRL-C
        print('Caught {} cleaning up and exiting.'.format(type(e).__name__))
    finally:
        # Send SIGTERM
        for task in ['ps', 'worker']:
            for process in processes[task]:
                if process.poll() is None:
                    process.terminate()

        # Wait for exit
        for task in ['ps', 'worker']:
            for process in processes[task]:
                process.wait()

    return 0

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser = configs.distributed.add_argparse_args(parser)

    parser.add_argument (
        '--launch_delay', type=float, default=0.0,
        help='Wait time between launching tasks in seconds.'
    )

    parser.add_argument (
        '--poll_delay', type=float, default=60.0,
        help='Wait time between checking tasks in seconds.'
    )

    FLAGS, CHILD_FLAGS = parser.parse_known_args()
    exit(main(FLAGS, CHILD_FLAGS))
