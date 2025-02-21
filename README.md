# hello-deepspeed

## Setup

- Modify config_sample and hostfile.sample as you need

```bash
echo config_sample >> ~/.ssh/config
cp hostfile.sample hostfile
```

## Install

```bash
pip install deepspeed
```

## (Optional) Port Range

- Run below if your environment have network/firewall configuration
- Modify `PORT_BEGIN`, `PORT_END` to your need

```
echo "net.ipv4.ip_local_port_range = PORT_BEGIN PORT_END" >> /etc/sysctl.conf
sysctl -p
```

## Run

```bash
deepspeed --hostfile hostfile --master_addr=YOUR_MASTER_ADDR --master_port=YOUR_MASTER_PORT train_hf.py
```
