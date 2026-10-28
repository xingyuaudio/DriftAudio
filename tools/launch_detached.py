"""Linux: launch a supervised command detached from VPN/SSH, with persistent status."""
import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log-dir',required=True); p.add_argument('--supervise',action='store_true')
    p.add_argument('command',nargs=argparse.REMAINDER); a=p.parse_args()
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command:p.error('supply a command after --')
    log=Path(a.log_dir).resolve()
    if not a.supervise:
        log.mkdir(parents=True,exist_ok=False)
        with (log/'supervisor.log').open('x') as stream:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--supervise','--log-dir',str(log),'--',*command],
   stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=child.pid,log_dir=str(log))))
        return
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    def terminate(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate)
    (log/'pid').write_text(str(os.getpid())+'\n')
    info=dict(pid=os.getpid(),pgid=os.getpgrp(),sid=os.getsid(0),cwd=os.getcwd(),command=command,
              gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),started=datetime.datetime.now().astimezone().isoformat(),
              log=str(log/'supervisor.log'), exit_code=str(log/'exit_code'))
    (log/'run_info.json').write_text(json.dumps(info,indent=2)+'\n')
    code=1
    try:code=subprocess.call(command,stdin=subprocess.DEVNULL)
    except SystemExit as e:code=e.code
    finally:
        (log/'exit_code').write_text(str(code)+'\n')
        (log/'finished_at').write_text(datetime.datetime.now().astimezone().isoformat()+'\n')
    sys.exit(code)


if __name__=='__main__':main()
