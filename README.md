<h1>
selfchatbot
</h1>

<p>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow"/></a>
</p>

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

<h2>Table of Contents</h2>

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
- [Parameters JSON](#parameters-json)
  - [Sample `parameters.json`](#sample-parametersjson)
  - [Field Descriptions](#field-descriptions)
- [Usage](#usage)
  - [Preprocessing](#preprocessing)
  - [Finetuning](#finetuning)
  - [Playing with the Chat Model](#playing-with-the-chat-model)
- [Authors](#authors)

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
- **`Name`**: A descriptive, human-readable name for the dataset. This can be any string that helps you identify the dataset (e.g., `RedditChats`, `Discord_Data`).

#### Examples

- `Dataset_1_Discord_Data`, here `1` is the ID and `Discord_Data` is the name
- `Dataset_2_RedditChats`, here `2` is the ID and `RedditChats` is the name

### Contents of Each Dataset Folder

1. **`parameters.json`**
    - Each dataset folder must contain a `parameters.json` file, specifying fine-tuning parameters. See the [Parameters](#parameters-json) section for more details.

2. Sub-Folders for Message Formats
    - Each dataset folder contains sub-folders named after the format of the .txt files they contain. Examples of sub-folder names include:

        - **DiscordChatExporter** for data exported by DiscordChatExporter.

3. .txt Files
    - Each sub-folder contains .txt files representing the messages. These files must conform to the format associated with their sub-folder name

### Example Directory Layout
Here's an example directory layout for a dataset:
```
selfChatBot_raw/
├── Dataset_1_Discord_Data/
│   ├── parameters.json
│   ├── DiscordChatExporter/
│   │   ├── channel1.txt
│   │   ├── channel2.txt
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


## Parameters JSON

Each dataset folder in `selfChatBot_raw` must include a `parameters.json` file that specifies the parameters for fine-tuning. Below is an example and detailed explanation of the structure and its fields.

### Sample `parameters.json`

```json
{
    "model": "gpt2-large",
    "type_fine_tune": "lora",
    "max_length": 1024, // Defaults to 1024 if not specified
    "preprocessor_data": {
        "DiscordChatExporter" : {
            "username": "your_username_here_without_@"
        }
    }
}
```

### Field Descriptions
1. **`model`**
   - **Description:** Specifies the base model to use for fine-tuning.
   - **Example:** `"gpt2-large"`, `"gpt2-xl"`, `"EleutherAI/gpt-neo-1.3B"`
2. **`type_fine_tune`**
    - **Description:** Defines the fine-tuning method.
    - **Allowed Values:**
      - **`lora`**: Use LoRA fine-tuning (Low Rank Adaptation).
      - **`qlora`**: Use Quantized LoRA.
      - **`fine_tune`**: Full model fine-tuning.
3. **`max_length`**
    - **Description:** The maximum sequence length for training and inference.
    - **Example:** 1024 (recommended for GPT-2 models).
4. **`preprocessor_data`**
    - **Description:** A nested field containing dataset-specific preprocessing parameters.
    - **Structure:**
      - The keys are the names of sub-folder formats (e.g., `DiscordChatExporter`, `SlackExporter`).
      - Each key maps to an object with format-specific settings.

    **Example for `DiscordChatExporter`:**
    ```json
    "preprocessor_data": {
        "DiscordChatExporter": {
            "username": "your_username_here_without_@"
        }
    }
    ```
    - **Field:** **`username`**
      - **Description:**  Your Discord username without the @.
      - **Example:** `"john_doe"`


<h3>Notes</h3>

- Make sure all fields are correctly defined; missing or invalid values can cause errors during fine-tuning.
- The preprocessor_data field is optional but must be included if specific preprocessing is required for the dataset's format.
- For fine-tuning with different methods (lora, qlora, or finetune), ensure the base model and parameters are compatible with the chosen method

## Usage
Before using selfchatbot make sure you have checked out [Installation](#installation), [Environment Variables](#environment-variables), [Data Folder Structure](#dataset-folder-structure), and [Parameters JSON](#parameters-json). For a quick start, see [SampleEnvironment](https://github.com/kagamiAL/selfchatbot/tree/main/SampleEnvironment).

### Preprocessing
You must preprocess your data before moving on to finetuning

To preprocess a dataset, use the command-line tool `selfChatBot_preprocess`. This command processes data from various sources within a dataset folder and prepares it for fine-tuning.

<h4>Command:</h4>

```bash
selfChatBot_preprocess -d <Dataset_ID>
```
- **`-d <Dataset_ID>`**: The unique ID of the dataset to preprocess. This corresponds to the ID in the dataset folder name (Dataset_{ID}_{Name}).

<h4>Example:</h4>

```bash
selfChatBot_preprocess -d 1
```
This example preprocesses the dataset located in `selfChatBot_raw/Dataset_1_Discord_Data`.

<h4>Notes</h4>

- The preprocessed data will be saved in the `selfChatBot_preprocessed` directory.
- Refer to the [Data Folder Structure](#dataset-folder-structure) and [Parameters JSON](#parameters-json) section for details on preparing datasets before fine-tuning.

### Finetuning
You must have preprocessed your data before moving on to finetuning

To fine-tune a model on a dataset, use the command-line tool `selfChatBot_train`. This command trains the model using preprocessed data from the specified dataset.

<h4>Command</h4>

```bash
selfChatBot_train -d <Dataset_ID>
```
- **`-d <Dataset_ID>`**: The unique ID of the dataset to fine-tune the model on. This corresponds to the ID in the dataset folder name (Dataset_{ID}_{Name}).

<h4>Notes</h4>

- The fine-tuning results (including the model weights) will be saved in the `selfChatBot_results` directory.
- Refer to the [Data Folder Structure](#dataset-folder-structure) and [Parameters JSON](#parameters-json) section for details on preparing datasets before fine-tuning.

### Playing with the Chat Model

To interact with the fine-tuned chat model, use the command-line tool `selfChatBot_play`. This command allows you to play with the model using either a session-based interaction or a prompt-based interaction.

<h4>Command</h4>

```bash
selfChatBot_play -d <Dataset_ID> [-t <interaction_type>] [-p <prompt>] [-mt <model_type>]
```

- **`-d <Dataset_ID>`**: The unique ID of the dataset to use for interaction. This corresponds to the **`ID`** in the dataset folder name (**`Dataset_{ID}_{Name}`**).
- **`-t <interaction_type>`**: (Optional) Specifies the type of interaction with the model.
  - **`session`**: Initiates an ongoing chat session.
  - **`prompt`**: Interacts with the model using a custom prompt.

  Default is **`session.
- **`-p <prompt>`**: (Required if **`-t prompt`** is selected) The custom prompt to use for the interaction when **`-t prompt`** is specified.
- **`-mt <model_type>`**: (Optional) Specifies which model to use for interaction.
  - **`best`**: Use the best-performing model (the one with the lowest validation loss).
  - **`final`**: Use the final model after training.

  Default is **`best`**.

<h4>Example</h4>

```bash
selfChatBot_play -d 1 -t prompt -p "Hello, how are you?"
```
This example uses the dataset **`Dataset_1_Discord_Data`** and sends the prompt `"Hello, how are you?"` to the model for a prompt-based interaction.

```bash
selfChatBot_play -d 1 -t session
```
This example initiates a session-based interaction with the model, using the dataset **`Dataset_1_Discord_Data`**.

<h4>Notes</h4>

- Ensure that the fine-tuned model is available in the **`selfChatBot_results`** directory.
- The model used for interaction can be the best model or the final model, depending on your preference.

## Authors
Alan Bach, bachalan330@gmail.com