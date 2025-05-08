# CodeTransBenchmark

### Install


### Dependencies
To install all Python dependencies, please execute the following command:
```
pip install -e codetrans
pip install -e codetransbenchmark
```

If unexpected errors occur, these can be due to the `comment-parser` dependency of `codetrans`.
Please refer to the Readme of `codetrans` for more information on this.

Additionally, the runtimes and compilers of all target languages have to be installed.
Because we use a Windows machine, we used the following versions of the compilers and runtimes to compile and test the translations: 
- C#: .NET 7.0.306 with MSBuild version 17.6.8+c70978d4d for .NET
- Go: GoLang 1.12.1
- Java: OpenJDK 17.0.10
- Python: Python 3.11.1, including installed packages such as Numpy and Pandas
- Rust: Rust 1.75.0

### Setup
Before running the pipeline you need to adjust the config in `codetransbenchmark/config/config.yaml` and in `codetrans/src/codetrans/codetrans_config.py` to your system.

We use llamafile and ollama as runtimes for inference with the LLMs.
To use GGUFs of the models, download the models and place them in a directory as specified in `codetrans/src/codetrans/codetrans_config.py`. 

### Dataset
The dataset contains samples from [CodeNet](https://github.com/IBM/Project_CodeNet), [AVATAR](https://github.com/wasiahmad/AVATAR), [Evalplus](https://github.com/evalplus/evalplus), [Geeks](https://github.com/yz1019117968/FSE-24-UniTrans), and our manually created BitHacks, which is based on samples from [Sean Eron Anderson](https://graphics.stanford.edu/~seander/bithacks.html) and [Brayoni](https://github.com/ianbrayoni/bithacks).
Translations of the EvalPlus and the Geeks datasets were not evaluated in our study as testing these translations is not yet implemented in CodeTransBenchmark.


Please unzip the `dataset.zip` file. After unzipping, you should see the following directory structure:

```
codetransbenchmark
├── dataset
    ├── avatar
    ├── bithacks
    ├── codenet
    ├── geeks
    ├── evalplus
├── ...
```

### Entrypoints
To use CodeTransBenchmark, use the `run_commands.py` in the `scripts` directory.

It provides the following CLI:

```
run_commands.py [-h] [-a ATTEMPT] [-opp OUTPUT_POST_PROCESSING] [-mini] [-over] [-d DATASET] [-e ENGINE] task model template

The positional arguments:
  task                  The task to perform. Valid values: 'translate', 'clean_generations', 'test', 'repair
  model                 Name of the model to use.
  template              Name of the prompt template to use.

options:
  -h, --help            show the help message
  -a ATTEMPT, --attempt ATTEMPT
                        The Attempt or Intertion number starting from 1.
  -opp OUTPUT_POST_PROCESSING, --output_post_processing OUTPUT_POST_PROCESSING
                        Which post-processing chain to apply on the LLM-generated output.
  -mini, --mini_benchmark
                        Whether to use the mini-benchmark instead of the datasets.
  -over, --overwrite    Whether to ignore the language pairs that were already executed and overwrite .
  -d DATASET, --dataset DATASET
                        The single dataset to use. Otherwise all datasets are used.
  -e ENGINE, --engine ENGINE
                        Name of the model engine to use. Valid values: 'llamafile', 'ollama', 'torch'. Note that there is only a basic implementation for using pytorch and the HuggingFace transformers library. Default: 'llamafile'.

```

1. Translate all datasets with llamafile and Mistral using the `controlled_md` prompt template.
```
python ./scripts/run_commands.py translate mistral controlled_md -a 1
```

2. Clean and post-process translations with the default strategy 'complex'. Other strategies can be selected via the optional argument -opp. The simpler post-processing chains are 'naive', 'remove_md', and 'controlled'. We recommend the default strategy as it showed the best generalisation across all models.
```
python ./scripts/run_commands.py clean_generations mistral controlled_md -a 1
```

Example with the `naive` approach: 
```
python ./scripts/run_commands.py clean_generations mistral controlled_md -a 1 -opp naive
```
The -opp option with the strategy identifier has to be given in all subsequent steps.

3. To compile and test translations, and generate feedback in form of error reports, you can run the following command:
```
python ./scripts/run_commands.py test mistral controlled_md -a 1
```

4. To repair unsuccessful translations of the first translation round, use the following command:
```
python ./scripts/run_commands.py repair mistral controlled_md -a 1
```


## Artefacts

The generated LLM outputs can be found in `output.zip`.
The test results and post-processing reports can be found in the `artefacts.zip`.

After unizpping, you should have the following structure:
```
codetransbenchmark
├── output
    ├── llamafile_codestral_controlled_md
    ├── ...
├── testresults
├── testresults_controlled_pp
├── testresults_naive_pp
├── testresults_remove_md_pp
├── ...
```




