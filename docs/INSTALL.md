# Installation

Requirements:
- [Git](https://git-scm.com/downloads)
- [Conda](https://www.anaconda.com/download)

*You should only need to do the steps in this section once. On UNIX, do these steps in terminal. On Windows, do them in Git Bash*

1. Clone repo from github
    - `git clone https://github.com/cirorandazzo/callback-analysis`
1. Clone submodules
    - `git submodule update --init --recursive`
1. Create conda env.
    - On Windows, do these steps in Anaconda Prompt.
    - Navigate to `callback-analysis` folder
    - `conda env create --file explicit_env.yaml`
