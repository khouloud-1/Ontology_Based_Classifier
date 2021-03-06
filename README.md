# Ontology_Based_Classifier
<html xmlns:v="urn:schemas-microsoft-com:vml"
xmlns:o="urn:schemas-microsoft-com:office:office"
xmlns:w="urn:schemas-microsoft-com:office:word"
xmlns:m="http://schemas.microsoft.com/office/2004/12/omml"
xmlns="http://www.w3.org/TR/REC-html40">

<head>
<meta http-equiv=Content-Type content="text/html; charset=utf-8">
<meta name=ProgId content=Word.Document>
<meta name=Generator content="Microsoft Word 12">
<meta name=Originator content="Microsoft Word 12">
<link rel=File-List href="GitHub_fichiers/filelist.xml">
<link rel=Edit-Time-Data href="GitHub_fichiers/editdata.mso">
<link rel=dataStoreItem href="GitHub_fichiers/item0001.xml"
target="GitHub_fichiers/props0002.xml">
<link rel=themeData href="GitHub_fichiers/themedata.thmx">
<link rel=colorSchemeMapping href="GitHub_fichiers/colorschememapping.xml">

</head>

<body lang=FR link=blue vlink=purple style='tab-interval:35.4pt'>

<div class=Section1>

<p class=MsoNormal align=center style='text-align:center;line-height:normal'><b><span
lang=EN-US style='font-size:22.0pt;mso-ansi-language:EN-US'>Ontology based
Feature Selection and Weighting for Text Classification<o:p></o:p></span></b></p>

<p class=MsoNormal style='text-align:justify;line-height:normal'><b><span
lang=EN-US style='mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></b></p>

<p class=MsoNormal style='text-align:justify;line-height:normal'><b><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>If you use software
from this page, please cite us as follows:<o:p></o:p></span></b></p>

<p class=MsoNormal style='text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>Djelloul BOUCHIHA,
Abdelghani BOUZIANE, Noureddine DOUMI and Mustafa JARRAR. Ontology based
Feature Selection and Weighting for Text Classification. Submitted (still under
review).<o:p></o:p></span></p>

<p class=MsoNormal style='text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='text-align:justify;line-height:normal'><b><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>If you need help,
contact </span></b><span lang=EN-US style='font-size:14.0pt;mso-ansi-language:
EN-US'>Djelloul BOUCHIHA<b>: <o:p></o:p></b></span></p>

<p class=MsoNormal style='text-align:justify;line-height:normal'><a
href="mailto:bouchiha@cuniv-naama.dz"><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>bouchiha@cuniv-naama.dz</span></a><span lang=EN-US
style='font-size:14.0pt;mso-ansi-language:EN-US'>; </span><a
href="mailto:bouchiha.dj@gmail.com"><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>bouchiha.dj@gmail.com</span></a><span lang=EN-US
style='font-size:14.0pt;mso-ansi-language:EN-US'>; </span><a
href="mailto:djelloul.bouchiha@univ-sba.dz"><span lang=EN-US style='font-size:
14.0pt;mso-ansi-language:EN-US'>djelloul.bouchiha@univ-sba.dz</span></a><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>; <o:p></o:p></span></p>

<p class=MsoNormal style='text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='text-align:justify;line-height:normal'><b><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>Before using any
software from this page, please follow carefully the following notes:<o:p></o:p></span></b></p>

<p class=MsoNormal style='text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>First you have to
download the BBC dataset from </span><a
href="http://mlg.ucd.ie/datasets/bbc.html"><span style='font-size:14.0pt'>http://mlg.ucd.ie/datasets/bbc.html</span></a><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>. Once downloaded,
you get </span><span style='font-size:14.0pt;font-family:"Segoe UI","sans-serif";
color:#212529;background:white'>2225</span><span lang=EN-US style='font-size:
14.0pt;mso-ansi-language:EN-US'> pre-classified English documents in one csv
file (</span><a href="Used_Corpus/BBC_Datasets/BBC_dataset.csv"><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>BBC_dataset.csv</span></a><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>)<o:p></o:p></span></p>

<p class=MsoNormal style='text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>The downloaded
dataset must be saved in the directory <span style='background:yellow;
mso-highlight:yellow'>'Used_Corpus/BBC_Datasets/</span></span><span lang=EN-US
style='font-size:14.0pt;mso-bidi-font-family:Calibri;mso-bidi-theme-font:minor-latin;
background:yellow;mso-highlight:yellow;mso-ansi-language:EN-US'>'</span><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>. In each
classifier code, the corpus path is mentioned as follows:<o:p></o:p></span></p>

<p class=MsoNormal style='text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>with open(<span
style='background:yellow;mso-highlight:yellow'>'Used_Corpus/BBC_Datasets/BBC_dataset.csv'</span>,
newline='', encoding='utf-8') as csvfile:<o:p></o:p></span></p>

<p class=MsoNormal style='text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='text-align:justify;line-height:normal'><b><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>Note that all
classifiers have been implemented using the Scientific Python Development
Environment (Spyder IDE, version 4.1.5):<o:p></o:p></span></b></p>

<p class=MsoNormal style='text-align:justify;line-height:normal'><span
style='font-size:14.0pt;mso-fareast-language:FR;mso-no-proof:yes'>
 <img src="GitHub_fichiers/image001.png"/>
</span><span lang=EN-US style='font-size:14.0pt;mso-ansi-language:
EN-US'><o:p></o:p></span></p>

<p class=MsoNormal style='text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><b><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>Before running any of the classifier, you have to
install some additional Python packages:</span></b><span lang=EN-US
style='font-size:14.0pt;mso-ansi-language:EN-US'> <o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>Since we are using Anaconda environment including
Spyder (Python editor), then we add packages through Anaconda Prompt terminal:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span style='font-size:14.0pt;mso-fareast-language:
FR;mso-no-proof:yes'>
 <img src="GitHub_fichiers/image002.png"/>
</span><span lang=EN-US style='font-size:14.0pt;mso-ansi-language:
EN-US'><o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>For the first time, check the already installed
packages with:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>C:\...&gt;<span style='background:yellow;mso-highlight:
yellow'>pip list</span><o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>Preprocessing task need to install <span
style='background:yellow;mso-highlight:yellow'>nltk</span>, <span
style='background:yellow;mso-highlight:yellow'>textblob</span> and <span
style='background:yellow;mso-highlight:yellow'>tashaphyne</span> packages. If
these packages don???t exist, you have to add them with:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>C:\...&gt;<span style='background:lime;mso-highlight:
lime'>pip install -U nltk</span><span style='background:yellow;mso-highlight:
yellow'><o:p></o:p></span></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>C:\...&gt;<span style='background:lime;mso-highlight:
lime'>pip install -U textblob</span><span style='background:yellow;mso-highlight:
yellow'><o:p></o:p></span></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>C:\...&gt;<span style='background:lime;mso-highlight:
lime'>pip install -U tashaphyne</span><o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>You have also to install <span style='background:yellow;
mso-highlight:yellow'>NLTK data</span> with:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>C:\Users\BOUCHIHA&gt;<span style='background:lime;
mso-highlight:lime'>python -m nltk.downloader popular</span><o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>BOW feature extraction technique needs <span
style='background:yellow;mso-highlight:yellow'>gensim</span> package. If this
package doesn???t exist, simply run:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal;tab-stops:45.8pt 91.6pt 137.4pt 183.2pt 229.0pt 274.8pt 320.6pt 366.4pt 412.2pt 458.0pt 503.8pt 549.6pt 595.4pt 641.2pt 687.0pt 732.8pt'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>C:\...&gt;<span
style='background:lime;mso-highlight:lime'>pip install -U gensim</span><o:p></o:p></span></p>

<p class=MsoNormal style='text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
background:yellow;mso-highlight:yellow;mso-ansi-language:EN-US'>TFIDF</span><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'> feature extraction
technique needs <span style='background:yellow;mso-highlight:yellow'>numpy</span>
package. If this package doesn???t exist, you have to install it:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal;tab-stops:45.8pt 91.6pt 137.4pt 183.2pt 229.0pt 274.8pt 320.6pt 366.4pt 412.2pt 458.0pt 503.8pt 549.6pt 595.4pt 641.2pt 687.0pt 732.8pt'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>C:\...&gt;<span
style='background:lime;mso-highlight:lime'>pip install -U numpy</span><o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>Don???t forget to update scikit-learn, the package
needed to implement machine learning algorithms, with the following command:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>C:\...&gt;<span style='background:lime;mso-highlight:
lime'>pip install -U scikit-learn</span><o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='text-align:justify;line-height:normal'><b><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>Next, you will find
4 classifiers named as follows:<o:p></o:p></span></b></p>

<p class=MsoNormal style='text-align:justify;line-height:normal'><a
href="01-Pre_OntologyBased_SVC.py"><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>01-Pre_OntologyBased_SVC.py</span></a><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p></o:p></span></p>

<p class=MsoNormal style='text-align:justify;line-height:normal'><a
href="02-Pre_BOW_SVC.py"><span lang=EN-US style='font-size:14.0pt;mso-ansi-language:
EN-US'>02-Pre_BOW_SVC.py</span></a><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'><o:p></o:p></span></p>

<p class=MsoNormal style='text-align:justify;line-height:normal'><a
href="03-Pre_TFIDF_SVC.py"><span lang=EN-US style='font-size:14.0pt;mso-ansi-language:
EN-US'>03-Pre_TFIDF_SVC.py</span></a><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'><o:p></o:p></span></p>

<p class=MsoNormal style='text-align:justify;line-height:normal'><a
href="04-Pre_Doc2Vec_SVC.py"><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>04-Pre_Doc2Vec_SVC.py</span></a><span lang=EN-US
style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p></o:p></span></p>

<p class=MsoNormal style='text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='text-align:justify;line-height:normal'><b><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>Next are some error
and warning messages that you may meet when dealing with our classifiers:<o:p></o:p></span></b></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>If you get the following warning message:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>UserWarning: The gensim.similarities.levenshtein
submodule is disabled, because the optional Levenshtein package
&lt;https://pypi.org/project/python-Levenshtein/&gt; is unavailable. Install
Levenhstein (e.g. `pip install python-Levenshtein`) to suppress this
warning.<span style='mso-spacerun:yes'>?? </span>warnings.warn(msg)<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:6.0pt;margin-right:0cm;margin-bottom:0cm;
margin-left:0cm;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-top:6.0pt;margin-right:0cm;margin-bottom:0cm;
margin-left:0cm;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>Then, try to add <span
style='background:yellow;mso-highlight:yellow'>python-Levenshtein</span>
package as follows:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>C:\...&gt;conda install -c conda-forge
python-levenshtein</span><span lang=EN-US style='font-size:14.0pt;mso-ansi-language:
EN-US'><o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal;tab-stops:45.8pt 91.6pt 137.4pt 183.2pt 229.0pt 274.8pt 320.6pt 366.4pt 412.2pt 458.0pt 503.8pt 549.6pt 595.4pt 641.2pt 687.0pt 732.8pt'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<div style='mso-element:para-border-div;border:none;border-bottom:solid windowtext 1.0pt;
mso-border-bottom-alt:solid windowtext .75pt;padding:0cm 0cm 1.0pt 0cm'>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal;tab-stops:45.8pt 91.6pt 137.4pt 183.2pt 229.0pt 274.8pt 320.6pt 366.4pt 412.2pt 458.0pt 503.8pt 549.6pt 595.4pt 641.2pt 687.0pt 732.8pt;
border:none;mso-border-bottom-alt:solid windowtext .75pt;padding:0cm;
mso-padding-alt:0cm 0cm 1.0pt 0cm'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

</div>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal;tab-stops:45.8pt 91.6pt 137.4pt 183.2pt 229.0pt 274.8pt 320.6pt 366.4pt 412.2pt 458.0pt 503.8pt 549.6pt 595.4pt 641.2pt 687.0pt 732.8pt'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal;tab-stops:45.8pt 91.6pt 137.4pt 183.2pt 229.0pt 274.8pt 320.6pt 366.4pt 412.2pt 458.0pt 503.8pt 549.6pt 595.4pt 641.2pt 687.0pt 732.8pt'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>If you get the following warning message:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>UndefinedMetricWarning: Precision and F-score are
ill-defined and being set to 0.0 in labels with no predicted samples. Use
`zero_division` parameter to control this behavior.<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'><span style='mso-spacerun:yes'>??
</span>_warn_prf(average, modifier, msg_start, len(result))<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:6.0pt;margin-right:0cm;margin-bottom:0cm;
margin-left:0cm;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-top:6.0pt;margin-right:0cm;margin-bottom:0cm;
margin-left:0cm;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>This warning
message disappears once <span style='background:yellow;mso-highlight:yellow'>the
zero_division</span> parameter is set to <span style='background:yellow;
mso-highlight:yellow'>0</span> or <span style='background:yellow;mso-highlight:
yellow'>1</span>. For example:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>classification_report(y_true, y_pred,
target_names=target_names, zero_division=0)</span><span lang=EN-US
style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p></o:p></span></p>

<div style='mso-element:para-border-div;border:none;border-bottom:solid windowtext 1.0pt;
mso-border-bottom-alt:solid windowtext .75pt;padding:0cm 0cm 1.0pt 0cm'>

<p class=MsoNormal style='text-align:justify;line-height:normal;border:none;
mso-border-bottom-alt:solid windowtext .75pt;padding:0cm;mso-padding-alt:0cm 0cm 1.0pt 0cm'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='text-align:justify;line-height:normal;border:none;
mso-border-bottom-alt:solid windowtext .75pt;padding:0cm;mso-padding-alt:0cm 0cm 1.0pt 0cm'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

</div>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>For <span style='background:yellow;mso-highlight:yellow'>LDA</span>
and <span style='background:yellow;mso-highlight:yellow'>QDA</span>
implementations, if you get the following warning message:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>C:\...\anaconda3\lib\site-packages\sklearn\discriminant_analysis.py:715:
UserWarning: Variables are collinear<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'><span style='mso-spacerun:yes'>??
</span>warnings.warn(&quot;Variables are collinear&quot;)<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:6.0pt;margin-right:0cm;margin-bottom:0cm;
margin-left:0cm;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-top:6.0pt;margin-right:0cm;margin-bottom:0cm;
margin-left:0cm;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>Open the file: <span
style='background:yellow;mso-highlight:yellow'>C:\...\anaconda3\lib\site-packages\sklearn\discriminant_analysis.py</span><o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>Remove or set as comment the following statements:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>rank = np.sum(S &gt; self.tol)<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>if rank &lt; n_features:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'><span style='mso-spacerun:yes'>??????????
</span>warnings.warn(&quot;Variables are collinear&quot;)<o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<div style='mso-element:para-border-div;border:none;border-bottom:solid windowtext 1.0pt;
mso-border-bottom-alt:solid windowtext .75pt;padding:0cm 0cm 1.0pt 0cm'>

<p class=MsoNormal style='text-align:justify;line-height:normal;border:none;
mso-border-bottom-alt:solid windowtext .75pt;padding:0cm;mso-padding-alt:0cm 0cm 1.0pt 0cm'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

</div>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>When predicting a text???s class, if you get the
following error message:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>Cannot center sparse matrices: pass `with_mean=False`
instead. See docstring for motivation and alternatives.</span><span lang=EN-US
style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:6.0pt;margin-right:0cm;margin-bottom:0cm;
margin-left:0cm;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-top:6.0pt;margin-right:0cm;margin-bottom:0cm;
margin-left:0cm;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>This can be
resolved by <span style='background:yellow;mso-highlight:yellow'>toarray</span>
conversion as follows:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>clf.predict(x_vec.toarray())</span><span lang=EN-US
style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p></o:p></span></p>

<div style='mso-element:para-border-div;border:none;border-bottom:solid windowtext 1.0pt;
mso-border-bottom-alt:solid windowtext .75pt;padding:0cm 0cm 1.0pt 0cm'>

<p class=MsoNormal style='text-align:justify;line-height:normal;border:none;
mso-border-bottom-alt:solid windowtext .75pt;padding:0cm;mso-padding-alt:0cm 0cm 1.0pt 0cm'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

</div>

<p class=MsoNormal style='text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>When executing <span style='background:yellow;
mso-highlight:yellow'>RadiusNeighborsClassifier</span>. If you get the
following error message:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>No neighbors found for test samples array([ ???],
dtype=int64), you can try using larger radius, giving a label for outliers, or
considering removing them from your dataset.</span><span lang=EN-US
style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:6.0pt;margin-right:0cm;margin-bottom:0cm;
margin-left:0cm;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-top:6.0pt;margin-right:0cm;margin-bottom:0cm;
margin-left:0cm;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>This can be
resolved by increasing the radius value as follows:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>RadiusNeighborsClassifier(radius = 40)</span><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<div style='mso-element:para-border-div;border:none;border-bottom:solid windowtext 1.0pt;
mso-border-bottom-alt:solid windowtext .75pt;padding:0cm 0cm 1.0pt 0cm'>

<p class=MsoNormal style='text-align:justify;line-height:normal;border:none;
mso-border-bottom-alt:solid windowtext .75pt;padding:0cm;mso-padding-alt:0cm 0cm 1.0pt 0cm'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

</div>

<p class=MsoNormal style='text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal;tab-stops:45.8pt 91.6pt 137.4pt 183.2pt 229.0pt 274.8pt 320.6pt 366.4pt 412.2pt 458.0pt 503.8pt 549.6pt 595.4pt 641.2pt 687.0pt 732.8pt'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>When executing <span
style='background:yellow;mso-highlight:yellow'>CategoricalNB</span> (Naive
Bayes), if you get the following error message:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>index .. is out of bounds for axis .. with size ..<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:6.0pt;margin-right:0cm;margin-bottom:0cm;
margin-left:0cm;margin-bottom:.0001pt;text-align:justify;line-height:normal;
tab-stops:45.8pt 91.6pt 137.4pt 183.2pt 229.0pt 274.8pt 320.6pt 366.4pt 412.2pt 458.0pt 503.8pt 549.6pt 595.4pt 641.2pt 687.0pt 732.8pt'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-top:6.0pt;margin-right:0cm;margin-bottom:0cm;
margin-left:0cm;margin-bottom:.0001pt;text-align:justify;line-height:normal;
tab-stops:45.8pt 91.6pt 137.4pt 183.2pt 229.0pt 274.8pt 320.6pt 366.4pt 412.2pt 458.0pt 503.8pt 549.6pt 595.4pt 641.2pt 687.0pt 732.8pt'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>This can be
resolved by increasing the value of the <span style='background:yellow;
mso-highlight:yellow'>min_categories</span> parameter of <span
style='background:yellow;mso-highlight:yellow'>CategoricalNB</span>, till the
error disappears. For example: <span style='background:yellow;mso-highlight:
yellow'>CategoricalNB(min_categories = 50)</span><o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal;tab-stops:45.8pt 91.6pt 137.4pt 183.2pt 229.0pt 274.8pt 320.6pt 366.4pt 412.2pt 458.0pt 503.8pt 549.6pt 595.4pt 641.2pt 687.0pt 732.8pt'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<div style='mso-element:para-border-div;border:none;border-bottom:solid windowtext 1.0pt;
mso-border-bottom-alt:solid windowtext .75pt;padding:0cm 0cm 1.0pt 0cm'>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal;tab-stops:45.8pt 91.6pt 137.4pt 183.2pt 229.0pt 274.8pt 320.6pt 366.4pt 412.2pt 458.0pt 503.8pt 549.6pt 595.4pt 641.2pt 687.0pt 732.8pt;
border:none;mso-border-bottom-alt:solid windowtext .75pt;padding:0cm;
mso-padding-alt:0cm 0cm 1.0pt 0cm'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

</div>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal;tab-stops:45.8pt 91.6pt 137.4pt 183.2pt 229.0pt 274.8pt 320.6pt 366.4pt 412.2pt 458.0pt 503.8pt 549.6pt 595.4pt 641.2pt 687.0pt 732.8pt'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal;tab-stops:45.8pt 91.6pt 137.4pt 183.2pt 229.0pt 274.8pt 320.6pt 366.4pt 412.2pt 458.0pt 503.8pt 549.6pt 595.4pt 641.2pt 687.0pt 732.8pt'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal;tab-stops:45.8pt 91.6pt 137.4pt 183.2pt 229.0pt 274.8pt 320.6pt 366.4pt 412.2pt 458.0pt 503.8pt 549.6pt 595.4pt 641.2pt 687.0pt 732.8pt'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal;tab-stops:45.8pt 91.6pt 137.4pt 183.2pt 229.0pt 274.8pt 320.6pt 366.4pt 412.2pt 458.0pt 503.8pt 549.6pt 595.4pt 641.2pt 687.0pt 732.8pt'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>Now, if you receive
the following error message:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>__init__() got an unexpected keyword argument
'min_categories'</span><span lang=EN-US style='font-size:14.0pt;mso-ansi-language:
EN-US'><o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:6.0pt;margin-right:0cm;margin-bottom:0cm;
margin-left:0cm;margin-bottom:.0001pt;text-align:justify;line-height:normal;
tab-stops:45.8pt 91.6pt 137.4pt 183.2pt 229.0pt 274.8pt 320.6pt 366.4pt 412.2pt 458.0pt 503.8pt 549.6pt 595.4pt 641.2pt 687.0pt 732.8pt'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-top:6.0pt;margin-right:0cm;margin-bottom:0cm;
margin-left:0cm;margin-bottom:.0001pt;text-align:justify;line-height:normal;
tab-stops:45.8pt 91.6pt 137.4pt 183.2pt 229.0pt 274.8pt 320.6pt 366.4pt 412.2pt 458.0pt 503.8pt 549.6pt 595.4pt 641.2pt 687.0pt 732.8pt'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>Then, you have to
update <span style='background:yellow;mso-highlight:yellow'>scikit-learn</span>
package (<span style='background:yellow;mso-highlight:yellow'>https://scikit-learn.org/stable/install.html</span>)<o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal;tab-stops:45.8pt 91.6pt 137.4pt 183.2pt 229.0pt 274.8pt 320.6pt 366.4pt 412.2pt 458.0pt 503.8pt 549.6pt 595.4pt 641.2pt 687.0pt 732.8pt'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<div style='mso-element:para-border-div;border:none;border-bottom:solid windowtext 1.0pt;
mso-border-bottom-alt:solid windowtext .75pt;padding:0cm 0cm 1.0pt 0cm'>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal;tab-stops:45.8pt 91.6pt 137.4pt 183.2pt 229.0pt 274.8pt 320.6pt 366.4pt 412.2pt 458.0pt 503.8pt 549.6pt 595.4pt 641.2pt 687.0pt 732.8pt;
border:none;mso-border-bottom-alt:solid windowtext .75pt;padding:0cm;
mso-padding-alt:0cm 0cm 1.0pt 0cm'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

</div>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal;tab-stops:45.8pt 91.6pt 137.4pt 183.2pt 229.0pt 274.8pt 320.6pt 366.4pt 412.2pt 458.0pt 503.8pt 549.6pt 595.4pt 641.2pt 687.0pt 732.8pt'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal;tab-stops:45.8pt 91.6pt 137.4pt 183.2pt 229.0pt 274.8pt 320.6pt 366.4pt 412.2pt 458.0pt 503.8pt 549.6pt 595.4pt 641.2pt 687.0pt 732.8pt'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>When executing <span style='background:yellow;
mso-highlight:yellow'>CategoricalNB</span>, <span style='background:yellow;
mso-highlight:yellow'>MultinomialNB</span> or <span style='background:yellow;
mso-highlight:yellow'>ComplementNB</span> (Naive Bayes), if you get the
following error message:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>Negative values in data passed to CategoricalNB (input
X)<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:6.0pt;margin-right:0cm;margin-bottom:0cm;
margin-left:0cm;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-top:6.0pt;margin-right:0cm;margin-bottom:0cm;
margin-left:0cm;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>That means <span
style='background:yellow;mso-highlight:yellow'>CategoricalNB</span> does not
admit negative vales, so you have to transform features by scaling each feature
to a given range (<span style='background:yellow;mso-highlight:yellow'>0</span>,
<span style='background:yellow;mso-highlight:yellow'>1</span> by default), by
using the following code:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>from sklearn import preprocessing<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>scaler1 = preprocessing.MinMaxScaler()<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>scaler1.fit(Xtrain)<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>Xtrain = scaler1.transform(Xtrain)<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>Scaler2 = preprocessing.MinMaxScaler()<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>Scaler2.fit(Xtest)<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>Xtest = scaler2.transform(Xtest)<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-left:21.3pt;text-align:justify;line-height:
normal'><span lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:
yellow;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<div style='mso-element:para-border-div;border:none;border-bottom:solid windowtext 1.0pt;
mso-border-bottom-alt:solid windowtext .75pt;padding:0cm 0cm 1.0pt 0cm'>

<p class=MsoNormal style='text-align:justify;line-height:normal;border:none;
mso-border-bottom-alt:solid windowtext .75pt;padding:0cm;mso-padding-alt:0cm 0cm 1.0pt 0cm'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

</div>

<p class=MsoNormal style='text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>When executing <span style='background:yellow;
mso-highlight:yellow'>GaussianProcessClassifier</span>, if you get the
following error message:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>C:\...\anaconda3\lib\site-packages\sklearn\gaussian_process\_gpc.py:472:
ConvergenceWarning: lbfgs failed to converge (status=2):<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>ABNORMAL_TERMINATION_IN_LNSRCH.<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>Increase the number of iterations (max_iter) or scale
the data as shown in:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'><span style='mso-spacerun:yes'>??????
</span>https://scikit-learn.org/stable/modules/preprocessing.html<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'><span style='mso-spacerun:yes'>??
</span>_check_optimize_result(&quot;lbfgs&quot;, opt_res)<o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>This can be resolved by scaling data as follows:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>from sklearn import preprocessing<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>scaler1 = preprocessing.StandardScaler().fit(Xtrain)<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>Xtrain = scaler1.transform(Xtrain)<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>scaler2 = preprocessing.StandardScaler().fit(Xtest)<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>Xtest = scaler2.transform(Xtest)<o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<div style='mso-element:para-border-div;border:none;border-bottom:solid windowtext 1.0pt;
mso-border-bottom-alt:solid windowtext .75pt;padding:0cm 0cm 1.0pt 0cm'>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal;border:none;mso-border-bottom-alt:solid windowtext .75pt;
padding:0cm;mso-padding-alt:0cm 0cm 1.0pt 0cm'><span lang=EN-US
style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

</div>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>When executing <span style='background:yellow;
mso-highlight:yellow'>MLPClassifier</span>, if you get the following warning
message:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>C:\...\anaconda3\lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:614:
ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and
the optimization hasn't converged yet.<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'><span style='mso-spacerun:yes'>?? </span>warnings.warn(<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:6.0pt;margin-right:0cm;margin-bottom:0cm;
margin-left:0cm;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-top:6.0pt;margin-right:0cm;margin-bottom:0cm;
margin-left:0cm;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>This can be
resolved by increasing the <span style='background:yellow;mso-highlight:yellow'>max_iter</span>
parameter value of <span style='background:yellow;mso-highlight:yellow'>MLPClassifier</span>.
For example:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>MLPClassifier(max_iter=700)<o:p></o:p></span></p>

<div style='mso-element:para-border-div;border:none;border-bottom:solid windowtext 1.0pt;
mso-border-bottom-alt:solid windowtext .75pt;padding:0cm 0cm 1.0pt 0cm'>

<p class=MsoNormal style='text-align:justify;line-height:normal;border:none;
mso-border-bottom-alt:solid windowtext .75pt;padding:0cm;mso-padding-alt:0cm 0cm 1.0pt 0cm'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

</div>

<p class=MsoNormal style='text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>When installing some packages, if you get the
following error message:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>ERROR: Could not install packages due to an
EnvironmentError: [WinError 5] Acc??s refus??:
'C:\\...\\anaconda3\\Lib\\site-packages\\...'<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>Consider using the `--user` option or check the
permissions.<o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>You can&nbsp;install the package for your user only,
like this:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:21.25pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
lang=EN-US style='font-size:14.0pt;background:yellow;mso-highlight:yellow;
mso-ansi-language:EN-US'>C:\...&gt;pip install &lt;package&gt; --user<o:p></o:p></span></p>

<p class=MsoNormal style='text-align:justify;line-height:normal'><b><span
lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>Or<o:p></o:p></span></b></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'>You can&nbsp;install the package as Administrator, by
following these steps:<o:p></o:p></span></p>

<p class=MsoListParagraphCxSpFirst style='margin-bottom:0cm;margin-bottom:.0001pt;
mso-add-space:auto;text-align:justify;text-indent:-18.0pt;line-height:normal;
mso-list:l3 level1 lfo4'><![if !supportLists]><span lang=EN-US
style='font-size:14.0pt;mso-bidi-font-family:Calibri;mso-bidi-theme-font:minor-latin;
mso-ansi-language:EN-US'><span style='mso-list:Ignore'>1.<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp; </span></span></span><![endif]><span
dir=LTR></span><span lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>Right
click on the Command Prompt icon<o:p></o:p></span></p>

<p class=MsoListParagraphCxSpLast style='margin-bottom:0cm;margin-bottom:.0001pt;
mso-add-space:auto;text-align:justify;text-indent:-18.0pt;line-height:normal;
mso-list:l3 level1 lfo4'><![if !supportLists]><span lang=EN-US
style='font-size:14.0pt;mso-bidi-font-family:Calibri;mso-bidi-theme-font:minor-latin;
mso-ansi-language:EN-US'><span style='mso-list:Ignore'>2.<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp; </span></span></span><![endif]><span
dir=LTR></span><span lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>Select
the option&nbsp;Run This Program As An Administrator<o:p></o:p></span></p>

<p class=MsoNormal style='margin-top:0cm;margin-right:0cm;margin-bottom:0cm;
margin-left:18.0pt;margin-bottom:.0001pt;text-align:justify;line-height:normal'><span
style='font-size:14.0pt;mso-fareast-language:FR;mso-no-proof:yes'>
 <img src="GitHub_fichiers/image003.png"/>
</span><span lang=EN-US style='font-size:14.0pt;mso-ansi-language:
EN-US'><o:p></o:p></span></p>

<p class=MsoListParagraph style='margin-bottom:0cm;margin-bottom:.0001pt;
mso-add-space:auto;text-align:justify;text-indent:-18.0pt;line-height:normal;
mso-list:l3 level1 lfo4'><![if !supportLists]><span lang=EN-US
style='font-size:14.0pt;mso-bidi-font-family:Calibri;mso-bidi-theme-font:minor-latin;
mso-ansi-language:EN-US'><span style='mso-list:Ignore'>3.<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp; </span></span></span><![endif]><span
dir=LTR></span><span lang=EN-US style='font-size:14.0pt;mso-ansi-language:EN-US'>Run
the command&nbsp;<span style='background:yellow;mso-highlight:yellow'>pip
install -U &lt;package&gt;</span><o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0cm;margin-bottom:.0001pt;text-align:
justify;line-height:normal'><span lang=EN-US style='font-size:14.0pt;
mso-ansi-language:EN-US'><o:p>&nbsp;</o:p></span></p>

</div>

</body>

</html>
