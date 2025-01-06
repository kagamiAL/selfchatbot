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
- [Environment Variables](#environment-variables)
  - [Required Variables](#required-variables)
  - [Setting Environment Variables](#setting-environment-variables)
- [Dataset Folder Structure](#dataset-folder-structure)
  - [Folder Naming Convention](#folder-naming-convention)
    - [Examples](#examples)
  - [Contents of Each Dataset Folder](#contents-of-each-dataset-folder)
  - [Example Directory Layout](#example-directory-layout)
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

## Environment Variables

selfchatbot requires several environment variables to be set for proper functionality. These variables determine the paths for storing raw data, preprocessed data, and results. You can declare these variables directly in your environment or use a `.env` file (built-in support is provided).

### Required Variables

- **`selfChatBot_raw`**: Path to the directory containing raw datasets.
- **`selfChatBot_preprocessed`**: Path to the directory for storing preprocessed datasets.
- **`selfChatBot_results`**: Path to the directory where results will be saved.

### Setting Environment Variables

1. **Using a `.env` File**
   Create a `.env` file in the root directory of selfchatbot and add the following lines:

    ```dotenv
    selfChatBot_raw=/path/to/raw/datasets
    selfChatBot_preprocessed=/path/to/preprocessed/datasets
    selfChatBot_results=/path/to/results
    ```

2. **Setting Variables Manually**

    On Linux/MacOS:
    ```dotenv
    export selfChatBot_raw=/path/to/raw/datasets
    export selfChatBot_preprocessed=/path/to/preprocessed/datasets
    export selfChatBot_results=/path/to/results
    ```

    On Windows (Command Prompt):
    ```dotenv
    set selfChatBot_raw=C:\path\to\raw\datasets
    set selfChatBot_preprocessed=C:\path\to\preprocessed\datasets
    set selfChatBot_results=C:\path\to\results
    ```

<h3>Notes</h3>

- Ensure that the paths you provide are absolute paths for consistency.
- The project will automatically detect and load the .env file if it exists.
- For large datasets, ensure the directories have sufficient storage capacity.


## Dataset Folder Structure

Datasets in the `selfChatBot_raw` directory must adhere to a specific folder structure to ensure proper functionality. Below are the details for organizing datasets:

### Folder Naming Convention

Each dataset folder should follow the naming template:

Dataset_`ID`_`Name`

- **`ID`**: A unique integer identifier for the dataset. This ensures each dataset is distinct (e.g., `1`, `2`, `42`).
- **`Name`**: A descriptive, human-readable name for the dataset. This can be any string that helps you identify the dataset (e.g., `RedditChats`, `DiscordLogs`).

#### Examples

- `Dataset_1_DiscordLogs`, here `1` is the ID and `DiscordLogs` is the name
- `Dataset_2_RedditChats`, here `2` is the ID and `RedditChats` is the name

### Contents of Each Dataset Folder

1. **`parameters.json`**
    - Each dataset folder must contain a `parameters.json` file, specifying fine-tuning parameters. See the [Parameters](#parameters) section for more details.

2. Sub-Folders for Message Formats
    - Each dataset folder contains sub-folders named after the format of the .txt files they contain. Examples of sub-folder names include:

        - **DiscordChatExporter** for data exported by DiscordChatExporter.

3. .txt Files
    - Each sub-folder contains .txt files representing the messages. These files must conform to the format associated with their sub-folder name

### Example Directory Layout
Here's an example directory layout for a dataset:
```
selfChatBot_raw/
├── Dataset_1_DiscordLogs/
│   ├── parameters.json
│   ├── DiscordChatExporter/
│   │   ├── channel1.txt
│   │   ├── channel2.txt
│   └── GenericFormat/
│       ├── misc.txt
├── Dataset_2_RedditChats/
│   ├── parameters.json
│   ├── RedditFormat/
│       ├── thread1.txt
│       ├── thread2.txt
```
For more clarity, you can look at the [SampleEnvironment](https://github.com/kagamiAL/selfchatbot/tree/main/SampleEnvironment) folder.

<h3>Notes</h3>

- Use unique integers for ID to avoid conflicts.
- Choose meaningful names for Name to easily identify datasets.
- Ensure the parameters.json file is present in every dataset folder and contains all required parameters.
- Sub-folder names must reflect the format of their .txt files for clarity and proper processing.

## Authors
Alan Bach, bachalan330@gmail.com