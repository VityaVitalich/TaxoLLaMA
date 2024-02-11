================================================================================
                              SEMEVAL-2016 TASK 13
                   TExEval-2: Taxonomy Extraction Evaluation
		  Georgeta Bordea, Els Lefever, Paul Buitelaar  
================================================================================
		    http://alt.qcri.org/semeval2016/task13/
================================================================================
TExEval-2_testdata_EN_1.0
================================================================================

This package contains the following files:

  - README.txt: a textual description of this package
  - /gs_terms - folder containing lists of terms for four languages (i.e., /EN, /FR, /IT, /NL) and three domains (i.e., environment, food, science) from different sources (i.e., Eurovoc, Wordnet, and other combined sources)
  - /gs_taxo - folder containing gold standard taxonomies for the terms described above
  - [domain]_[source]_[language].terms - file containing a list of terms from a given [domain] and [language] extracted from [source]
  - [domain]_[source]_[language].taxo - file containing a gold standard taxonomy from a given [domain] and [language] extracted from [source]

================================================================================
FILE FORMATS
================================================================================

TERMS FILE FORMAT

Each file for a provided domain terminology is tab-separated as follows: 

    term_id <TAB> term

where:
  - term_id is a term identifier (numeric);
  - term is a domain term.

================================================================================

TAXONOMY FILE FORMAT

The input files format for the taxonomies (.taxo) is a
tab-separated fields:

relation_id <TAB> term <TAB> hypernym 

where:
- relation_id: is a relation identifier; 
- term: is a term of the taxonomy;
- hypernym: is a hypernym for the term. 

e.g

0<TAB>cat<TAB>animal
1<TAB>dog<TAB>animal
2<TAB>car<TAB>animal
....

The input files format for the system relation evaluation (.taxo.eval) is a
tab-separated fields:

relation_id <TAB> eval

where:
- relation_id: is a relation identifier; 
- eval: is an empty string if the relation is good, an "x" otherwise

e.g.
0<TAB>
1<TAB>
2<TAB>x
....


The input files format for the terminologies (.terms) is a
tab-separated fields:

term_id <TAB> term

where:
- term_id: is a term identifier; 
- term: is a domain term.

================================================================================

For more information please visit:

  - task homepage:  http://alt.qcri.org/semeval2016/task13/
  - task google group: https://groups.google.com/d/forum/semeval-texeval


