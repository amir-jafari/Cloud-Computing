# Cloud Computing Software Setup Checklist

A comprehensive guide to setting up your development environment for cloud computing coursework.

---

## Getting Started

- [ ] **Introduction** - Remote connections, SSH, terminals, cloud platforms, servers, VMs, engines, and AMI concepts

### Architecture Overview

```
┌─────────────────────────────────┐              ┌─────────────────────────────────┐
│       LOCAL MACHINE             │              │      CLOUD SERVER               │
│                                 │              │   (AWS EC2 / GCP Compute)       │
├─────────────────────────────────┤              ├─────────────────────────────────┤
│                                 │              │                                 │
│  ┌──────────────────────┐       │              │  ┌──────────────────────┐       │
│  │   Terminal Client    │       │              │  │    SSH Daemon        │       │
│  │  (MobaXterm/Tabby)   │       │              │  │   (Port 22)          │       │
│  └──────────────────────┘       │              │  └──────────────────────┘       │
│           │                     │              │           │                     │
│           │                     │              │           │                     │
│  ┌────────▼─────────┐           │              │  ┌────────▼─────────┐           │
│  │  SSH Client      │           │              │  │  Linux Shell     │           │
│  │  (Private Key)   │◄──────────┼──────────────┼─►│  (Public Key)    │           │
│  └──────────────────┘           │   Encrypted  │  └──────────────────┘           │
│                                 │   Connection │                                 │
│  ┌──────────────────────┐       │   (SSH/TLS)  │  ┌──────────────────────┐       │
│  │   PyCharm IDE        │       │              │  │  Python Interpreter  │       │
│  │  (Local Editor)      │◄──────┼──────────────┼─►│  Libraries & Code    │       │
│  └──────────────────────┘       │   SFTP/SSH   │  └──────────────────────┘       │
│           │                     │              │           │                     │
│  ┌────────▼─────────┐           │              │  ┌────────▼─────────┐           │
│  │  Git Client      │           │              │  │   GPU/CPU         │          │
│  │  (Version Ctrl)  │           │              │  │   (nvidia-smi)    │          │
│  └──────────────────┘           │              │  └──────────────────┘           │
│                                 │              │                                 │
└─────────────────────────────────┘              └─────────────────────────────────┘

Connection Flow:
1. Launch cloud instance → 2. Generate SSH keys → 3. Connect via terminal
4. Configure environment → 5. Setup PyCharm remote → 6. Deploy & debug code
```

---

## Cloud Platform Fundamentals

- [ ] **What is a Dashboard?** - Navigate cloud provider interfaces
- [ ] **Going Over the Document: How to Make an EC2 Instance with AWS** - Launch and configure EC2 instances
  - Change instance type - Modify CPU/RAM resources
  - Stop instances - Pause instances to save costs
  - Terminate instances - Permanently delete instances
  - IP address changes - Handle dynamic/static IP addresses
  - Elastic IP - Reserve static IP addresses
  - Security groups - Configure firewall rules
  - Key pairs - Manage SSH authentication

---

## Terminal Setup

- [ ] **Terminal (MobaXterm, Tabby)** - Install and configure terminal emulator
- [ ] **Useful Terminal Commands** - Master essential Linux commands
  - `ls` - List directory contents
  - `cd` - Change directory
  - `pwd` - Print working directory
  - `mkdir` - Create directory
  - `rm` - Remove files/directories
  - `cp` - Copy files/directories
  - `mv` - Move/rename files
  - `chmod` - Change file permissions
  - `chown` - Change file ownership
  - `nano` / `vim` - Text editors
  - `top` - Process monitoring
  - `htop` - Interactive process viewer
  - `nvidia-smi` - NVIDIA GPU monitoring
  - `nvtop` - Interactive GPU process viewer
  - `screen` - Keep processes running after disconnect
    - `screen -S mywork` - Start new session named "mywork"
    - `Ctrl+A, D` - Detach from session (keeps running)
    - `screen -ls` - List all screen sessions
    - `screen -r mywork` - Reattach to session "mywork"
    - `exit` - Close current screen session

---

## SSH Configuration

- [ ] **SSHing (AWS Mac and Windows)** - Establish secure shell connections

---

## File Transfer (SFTP)

- [ ] **SFTP File Management** - Transfer and browse remote files securely
  - MobaXterm - Built-in SFTP browser (left sidebar)
  - FileZilla - Standalone SFTP client for file transfers
  - PyCharm - Integrated remote file browser and sync


---

## PyCharm IDE Setup

- [ ] **PyCharm Setup** - Install and configure PyCharm Professional
- [ ] **PyCharm Folder Mapping** - Sync local and remote directories
- [ ] **PyCharm Remote Interpreter** - Execute code on remote Python environment
- [ ] **PyCharm SSH** - Connect PyCharm to remote server
- [ ] **PyCharm SSH Debugging** - Debug code on remote server
- [ ] **PyCharm Git** - Version control integration
- [ ] **PyCharm Projects and Coding** - Develop and test applications

---

## Google Cloud Platform (GCP)

- [ ] **GCP Coupons** - Activate free credits
- [ ] **GCP SSH Key Generator** - Create authentication keys
- [ ] **GCP Compute Engine** - Launch GCP virtual machines
- [ ] **GCP Cloud Storage** - Use cloud object storage
- [ ] **GCP and PyCharm** - Integrate GCP with PyCharm
- [ ] **GCP and PyCharm SSH** - Connect PyCharm to GCP instances

---

## Troubleshooting & Advanced Topics

- [ ] **Change IP Each Time the Instance Is Restarted** - Handle dynamic vs static IPs
- [ ] **Change Instance Type** - Modify compute resources

---

## Completion Notes

Once completed, you should have functional cloud instances on AWS/GCP, working SSH connections, PyCharm configured for remote development, and the ability to deploy and manage cloud-based applications.