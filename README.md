<h1>
selfchatbot
</h1>

<p>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow"/></a>
</p>

<h2>Table of Contents</h2>

- [What is selfchatbot?](#what-is-selfchatbot)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
- [Authors](#authors)


## What is selfchatbot?
**selfchatbot** is a LM fine tuning pipeline to fine tune
base HuggingFace LMs to talk like you.

**Disclaimer:** This project is a personal learning experience rather than
something designed for production level applications. You can see this in
many design choices I've made learning how to finetune LMs with HuggingFace.

This currently only works with Discord direct messages exported from
[DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter), but
other message formats can be supported.

Documentation on how to add your own message format support will be added
later. If you want to add support for a new message format now you can
look at the [Preprocessors](https://github.com/kagamiAL/selfchatbot/tree/main/src/Classes/Preprocessors).

## Installation

selfchatbot requires **Python 3.12** or higher. Follow these steps to install the package:

### Prerequisites

- **Python 3.12+**: Download the latest version from the [official Python website](https://www.python.org/downloads/).
- **pip**: Ensure you have the latest version of `pip` installed. Update it if necessary:

    ```bash
    python -m pip install --upgrade pip
    ```

<h3>1. Clone this repository</h3>

Clone the project repository to your local machine:
```bash
git clone https://github.com/kagamiAL/selfchatbot
cd selfchatbot
```

<h3>2. Create a Virtual Environment (Recommended)</h3>

Set up a virtual environment to manage dependencies:
```bash
python -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
```

<h3>3. Install the Package</h3>

Install the package along with its dependencies using `pip`:
```bash
pip install .
```

For more details about the installation process or dependencies, refer to the pyproject.toml.

## Authors
Alan Bach, bachalan330@gmail.com