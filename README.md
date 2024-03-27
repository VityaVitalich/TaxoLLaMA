# TaxoLLaMA

We present a unified model that is capable of solving taxonomy related tasks, described in our paper TaxoLLaMA: WordNet-based Model for Solving Multiple Lexical Sematic Tasks. Here you could find the instructions to reproduce all our results.

### Data

All of our data follows the same format, we have modified each test and train set from taxonomy-related tasks to follow this format, as it is handy to use and unified, which is crucial aspect.

Each train or test set presented with ```.pickle``` format, for the ease of python usage. It is a list of python dictionaries, each dictionary represents a single object. Each object has keys and corresponding values
- case: predict_hypernym or predict_multiple_hypernyms. It differs since we process one predicted hypernym and multiple of them differently. 
- parents: string representing hypernym for single hypernym case and list of strings in case of multiple
- children: string representing hyponym
- child_def: string representing definition for the hyponym

The data is availible directly in this repository with tar.gz archive.
#### Pre-training with WordNet

For pre-training with instructive WordNet dataset we should simply sample data in our format. The process of sampling and recreation of data is possible in ```DataConsructor/unified_model.ipynb``` notebook. As well we also publish created datasets and can be downloaded from folder named ```tax_instruct_wnet.tar.gz```.

#### Hypernym Discovery

Refactored for our format data could be downloaded from folder named ```data_hypernym_discovery.tar.gz```. Few-shots sampled from training set are stored in ```SemEval2018-Task9/few_shots``` directory

#### Taxonomy Enrichment

We have followed TaxoEnrich data and train/test split procedure and after converted data to our format that could be downloaded from folder named ```data_taxonomy_enrichment.tar.gz```.

#### Taxonomy Construction

We formatted all the pairs for ease of perplexity estimation. Nevertheless, we still need original data for examination. Everything could be downloaded from folder named ```data_taxonomy_construction.tar.gz```.

#### Lexical Entailment

We formatted the pairs for perplexity estimation, however initial data left in the same format and could be downloaded from folder named ```data_lexical_entailment.tar.gz```


### Training, Inference, Fine-tuning

To use private models, user is required to save the HF auth token in environment variable ```export HF_TOKEN=<your token>```. As well, you need to specify the home directory for caching HF, saving all outputs, trained models, etc. as ```export SAVING_DIR='/your/directory'```

To train the model in our setting one needs to simply run the ```train.py``` script with the configs, specified in ```configs/train.yml```. Every parameter is described there. It reports final metrics to WandB and prints to terminal.

To inference the pre-trained model later one can use the ```inference.py``` script with the configs in ```configs/inference.yml```. The final metrics are printed and predictions are saved in file.

To get the detailed metric report with examples one needs to run ```metrics.py``` script with the configs in ```configs/metrics.yml```. 

To fine tune the model for your dataset, you can simply treat pre-trained model as intiial checkpoint and run the ```train.py``` script with checkpoint, specified in configs.

### Perplexity Estimation

To estimate perplexity firstly the model should be saved with hugging face format and either uploaded to hugging face or stored locally. Those actions needed to properly upload model. 
Then one should run ```TExEval-2_testdata_1.2/est_ppl.py``` script with the configs ```TExEval-2_testdata_1.2/configs/ppl_est.yml```. It will save resulting perplexities in format of python dict where key is tuple (hyponym, hypernym) and value is perplexity. 

### Benchmarking model

#### Hypernym Discovery

One simply needs to inference model with desired set and the metrics will be outputed

#### Taxonomy Enrichmnet

Like in the Hypernym Discovery one needs first to inference on desired test set. However, to stay consisten with previous research, evaluation is performed with scaled MRR, that could be obtained with the precisely the same code as in TaxoEnrich in ```TaxonomyEnrichment/notebooks/taxoenrich_metrics.ipynb```

#### Taxonomy Construction

Once the perplexities are calculated, one could build the taxonomy by running ```TExEval-2_testdata_1.2/build_taxo.py``` with configs in ```TExEval-2_testdata_1.2/configs/build_taxo.yml```. The final metric will be outputed.

#### Lexical Entailment

Once the perplexities are calculated, one need to run ```LexicalEntailment/notebooks/ant_scorer.ipynb``` notebook with path to your test set and perplexities to calculate metrics for ANT dataset or run ```LexicalEntailment/notebooks/hyperlex_scorer.ipynb``` to obtain metrics for Hyperlex.

