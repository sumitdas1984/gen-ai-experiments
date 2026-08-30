arXiv:2408.09869v5 [cs.CL] 9 Dec 2024

![img-0.jpeg](img-0.jpeg)

# Docling Technical Report

Version 1.0

Christoph Auer Maksym Lysak Ahmed Nassar Michele Dolfi Nikolaos Livathinos  
Panos Vagenas Cesar Berrospi Ramis Matteo Omenetti Fabian Lindlbauer  
Kasper Dinkla Lokesh Mishra Yusik Kim Shubham Gupta Rafael Teixeira de Lima  
Valery Weber Lucas Morin Ingmar Meijer Viktor Kuropiatnyk Peter W. J. Staar

AI4K Group, IBM Research Rüschlikon, Switzerland

## Abstract

This technical report introduces *Docling*, an easy to use, self-contained, MIT-licensed open-source package for PDF document conversion. It is powered by state-of-the-art specialized AI models for layout analysis (DocLayNet) and table structure recognition (TableFormer), and runs efficiently on commodity hardware in a small resource budget. The code interface allows for easy extensibility and addition of new features and models.

## 1 Introduction

Converting PDF documents back into a machine-processable format has been a major challenge for decades due to their huge variability in formats, weak standardization and printing-optimized characteristic, which discards most structural features and metadata. With the advent of LLMs and popular application patterns such as retrieval-augmented generation (RAG), leveraging the rich content embedded in PDFs has become ever more relevant. In the past decade, several powerful document understanding solutions have emerged on the market, most of which are commercial software, cloud offerings [3] and most recently, multi-modal vision-language models. As of today, only a handful of open-source tools cover PDF conversion, leaving a significant feature and quality gap to proprietary solutions.

With *Docling*, we open-source a very capable and efficient document conversion tool which builds on the powerful, specialized AI models and datasets for layout analysis and table structure recognition we developed and presented in the recent past [12, 13, 9]. *Docling* is designed as a simple, self-contained python library with permissive license, running entirely locally on commodity hardware. Its code architecture allows for easy extensibility and addition of new features and models.

Docling Technical Report

1

Here is what Docling delivers today:

- Converts PDF documents to JSON or Markdown format, stable and lightning fast
- Understands detailed page layout, reading order, locates figures and recovers table structures
- Extracts metadata from the document, such as title, authors, references and language
- Optionally applies OCR, e.g. for scanned PDFs
- Can be configured to be optimal for batch-mode (i.e high throughput, low time-to-solution) or interactive mode (compromise on efficiency, low time-to-solution)
- Can leverage different accelerators (GPU, MPS, etc).

## 2 Getting Started

To use Docling, you can simply install the docling package from PyPI. Documentation and examples are available in our GitHub repository at github.com/DS4SD/docling. All required model assets¹ are downloaded to a local huggingface datasets cache on first use, unless you choose to pre-install the model assets in advance.

Docling provides an easy code interface to convert PDF documents from file system, URLs or binary streams, and retrieve the output in either JSON or Markdown format. For convenience, separate methods are offered to convert single documents or batches of documents. A basic usage example is illustrated below. Further examples are available in the Doclign code repository.

from docling.document_converter import DocumentConverter

source = "https://arxiv.org/pdf/2206.01062" # PDF path or URL
converter = DocumentConverter()
result = converter.convert_single(source)
print(result.render_as_markup()) # output: "## DocLayNet: A Large
Human-Annotated Dataset for Document-Layout Analysis [...]"

Optionally, you can configure custom pipeline features and runtime options, such as turning on or off features (e.g. OCR, table structure recognition), enforcing limits on the input document size, and defining the budget of CPU threads. Advanced usage examples and options are documented in the README file. Docling also provides a Dockerfile to demonstrate how to install and run it inside a container.

## 3 Processing pipeline

Docling implements a linear pipeline of operations, which execute sequentially on each given document (see Fig. 1). Each document is first parsed by a PDF backend, which retrieves the programmatic text tokens, consisting of string content and its coordinates on the page, and also renders a bitmap image of each page to support downstream operations. Then, the standard model pipeline applies a sequence of AI models independently on every page in the document to extract features and content, such as layout and table structures. Finally, the results from all pages are aggregated and passed through a post-processing stage, which augments metadata, detects the document language, infers reading-order and eventually assembles a typed document object which can be serialized to JSON or Markdown.

### 3.1 PDF backends

Two basic requirements to process PDF documents in our pipeline are a) to retrieve all text content and their geometric coordinates on each page and b) to render the visual representation of each page as it would appear in a PDF viewer. Both these requirements are encapsulated in Docling's PDF backend interface. While there are several open-source PDF parsing libraries available for python, we faced major obstacles with all of them for different reasons, among which were restrictive

¹ see huggingface.co/ds4sd/docling-models/

2

![img-1.jpeg](img-1.jpeg)

Figure 1: Sketch of Docling's default processing pipeline. The inner part of the model pipeline is easily customizable and extensible.

licensing (e.g. pymupdf [7]), poor speed or unrecoverable quality issues, such as merged text cells across far-apart text tokens or table columns (pypdfium, PyPDF) [15, 14].

We therefore decided to provide multiple backend choices, and additionally open-source a custom-built PDF parser, which is based on the low-level qpdf[4] library. It is made available in a separate package named docling-parse and powers the default PDF backend in Docling. As an alternative, we provide a PDF backend relying on pypdfium, which may be a safe backup choice in certain cases, e.g. if issues are seen with particular font encodings.

### 3.2 AI models

As part of Docling, we initially release two highly capable AI models to the open-source community, which have been developed and published recently by our team. The first model is a layout analysis model, an accurate object-detector for page elements [13]. The second model is TableFormer [12, 9], a state-of-the-art table structure recognition model. We provide the pre-trained weights (hosted on huggingface) and a separate package for the inference code as docling-ibm-models. Both models are also powering the open-access deepsearch-experience, our cloud-native service for knowledge exploration tasks.

### Layout Analysis Model

Our layout analysis model is an object-detector which predicts the bounding-boxes and classes of various elements on the image of a given page. Its architecture is derived from RT-DETR [16] and re-trained on DocLayNet [13], our popular human-annotated dataset for document-layout analysis, among other proprietary datasets. For inference, our implementation relies on the onnxruntime [5].

The Docling pipeline feeds page images at 72 dpi resolution, which can be processed on a single CPU with sub-second latency. All predicted bounding-box proposals for document elements are post-processed to remove overlapping proposals based on confidence and size, and then intersected with the text tokens in the PDF to group them into meaningful and complete units such as paragraphs, section titles, list items, captions, figures or tables.

### Table Structure Recognition

The TableFormer model [12], first published in 2022 and since refined with a custom structure token language [9], is a vision-transformer model for table structure recovery. It can predict the logical row and column structure of a given table based on an input image, and determine which table cells belong to column headers, row headers or the table body. Compared to earlier approaches, TableFormer handles many characteristics of tables, such as partial or no borderlines, empty cells, rows or columns, cell spans and hierarchy both on column-heading or row-heading level, tables with inconsistent indentation or alignment and other complexities. For inference, our implementation relies on PyTorch [2].

3

The Docling pipeline feeds all table objects detected in the layout analysis to the TableFormer model, by providing an image-crop of the table and the included text cells. TableFormer structure predictions are matched back to the PDF cells in post-processing to avoid expensive re-transcription text in the table image. Typical tables require between 2 and 6 seconds to be processed on a standard CPU, strongly depending on the amount of included table cells.

## OCR

Docling provides optional support for OCR, for example to cover scanned PDFs or content in bitmaps images embedded on a page. In our initial release, we rely on EasyOCR [1], a popular third-party OCR library with support for many languages. Docling, by default, feeds a high-resolution page image (216 dpi) to the OCR engine, to allow capturing small print detail in decent quality. While EasyOCR delivers reasonable transcription quality, we observe that it runs fairly slow on CPU (upwards of 30 seconds per page).

We are actively seeking collaboration from the open-source community to extend Docling with additional OCR backends and speed improvements.

### 3.3 Assembly

In the final pipeline stage, Docling assembles all prediction results produced on each page into a well-defined datatype that encapsulates a converted document, as defined in the auxiliary package docling-core. The generated document object is passed through a post-processing model which leverages several algorithms to augment features, such as detection of the document language, correcting the reading order, matching figures with captions and labelling metadata such as title, authors and references. The final output can then be serialized to JSON or transformed into a Markdown representation at the users request.

### 3.4 Extensibility

Docling provides a straight-forward interface to extend its capabilities, namely the model pipeline. A model pipeline constitutes the central part in the processing, following initial document parsing and preceding output assembly, and can be fully customized by sub-classing from an abstract base-class (BaseModelPipeline) or cloning the default model pipeline. This effectively allows to fully customize the chain of models, add or replace models, and introduce additional pipeline configuration parameters. To use a custom model pipeline, the custom pipeline class to instantiate can be provided as an argument to the main document conversion methods. We invite everyone in the community to propose additional or alternative models and improvements.

Implementations of model classes must satisfy the python Callable interface. The __call__ method must accept an iterator over page objects, and produce another iterator over the page objects which were augmented with the additional features predicted by the model, by extending the provided PagePredictions data model accordingly.

## 4 Performance

In this section, we establish some reference numbers for the processing speed of Docling and the resource budget it requires. All tests in this section are run with default options on our standard test set distributed with Docling, which consists of three papers from arXiv and two IBM Redbooks, with a total of 225 pages. Measurements were taken using both available PDF backends on two different hardware systems: one MacBook Pro M3 Max, and one bare-metal server running Ubuntu 20.04 LTS on an Intel Xeon E5-2690 CPU. For reproducibility, we fixed the thread budget (through setting OMP_NUM_THREADS environment variable) once to 4 (Docling default) and once to 16 (equal to full core count on the test hardware). All results are shown in Table 1.

If you need to run Docling in very low-resource environments, please consider configuring the pypdfium backend. While it is faster and more memory efficient than the default docling-parse backend, it will come at the expense of worse quality results, especially in table structure recovery.

Establishing GPU acceleration support for the AI models is currently work-in-progress and largely untested, but may work implicitly when CUDA is available and discovered by the onnxruntime and

4

torch runtimes backing the Docling pipeline. We will deliver updates on this topic at in a future version of this report.

Table 1: Runtime characteristics of Docling with the standard model pipeline and settings, on our test dataset of 225 pages, on two different systems. OCR is disabled. We show the time-to-solution (TTS), computed throughput in pages per second, and the peak memory used (resident set size) for both the Docling-native PDF backend and for the pypdfium backend, using 4 and 16 threads.

|  CPU | Thread budget | native backend |   |   | pypdfium backend  |   |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
|   |   |  TTS | Pages/s | Mem | TTS | Pages/s | Mem  |
|  Apple M3 Max (16 cores) | 4 | 177 s | 1.27 | 6.20 GB | 103 s | 2.18 | 2.56 GB  |
|   |  16 | 167 s | 1.34 |   | 92 s | 2.45  |   |
|  Intel(R) Xeon E5-2690 (16 cores) | 4 | 375 s | 0.60 | 6.16 GB | 239 s | 0.94 | 2.42 GB  |
|   |  16 | 244 s | 0.92 |   | 143 s | 1.57  |   |

## 5 Applications

Thanks to the high-quality, richly structured document conversion achieved by Docling, its output qualifies for numerous downstream applications. For example, Docling can provide a base for detailed enterprise document search, passage retrieval or classification use-cases, or support knowledge extraction pipelines, allowing specific treatment of different structures in the document, such as tables, figures, section structure or references. For popular generative AI application patterns, such as retrieval-augmented generation (RAG), we provide quackling, an open-source package which capitalizes on Docling's feature-rich document output to enable document-native optimized vector embedding and chunking. It plugs in seamlessly with LLM frameworks such as LlamaIndex [8]. Since Docling is fast, stable and cheap to run, it also makes for an excellent choice to build document-derived datasets. With its powerful table structure recognition, it provides significant benefit to automated knowledge-base construction [11, 10]. Docling is also integrated within the open IBM data prep kit [6], which implements scalable data transforms to build large-scale multi-modal training datasets.

## 6 Future work and contributions

Docling is designed to allow easy extension of the model library and pipelines. In the future, we plan to extend Docling with several more models, such as a figure-classifier model, an equation-recognition model, a code-recognition model and more. This will help improve the quality of conversion for specific types of content, as well as augment extracted document metadata with additional information. Further investment into testing and optimizing GPU acceleration as well as improving the Docling-native PDF backend are on our roadmap, too.

We encourage everyone to propose or implement additional features and models, and will gladly take your inputs and contributions under review. The codebase of Docling is open for use and contribution, under the MIT license agreement and in alignment with our contributing guidelines included in the Docling repository. If you use Docling in your projects, please consider citing this technical report.

## References

[1] J. AI. Easyocr: Ready-to-use ocr with 80+ supported languages. https://github.com/JaidedAI/EasyOCR, 2024. Version: 1.7.0.

[2] J. Ansel, E. Yang, H. He, N. Gimelshein, A. Jain, M. Voznesensky, B. Bao, P. Bell, D. Berard, E. Burovski, G. Chauhan, A. Chourdia, W. Constable, A. Desmaison, Z. DeVito, E. Ellison, W. Feng, J. Gong, M. Gschwind, B. Hirsh, S. Huang, K. Kalambarkar, L. Kirsch, M. Lazos, M. Lezcano, Y. Liang, J. Liang, Y. Lu, C. Luk, B. Maher, Y. Pan, C. Puhrsch, M. Reso, M. Saroufim, M. Y. Siraichi, H. Suk, M. Suo, P. Tillet, E. Wang, X. Wang, W. Wen, S. Zhang, X. Zhao, K. Zhou, R. Zou, A. Mathews, G. Chanan, P. Wu, and S. Chintala. Pytorch 2: Faster

5

machine learning through dynamic python bytecode transformation and graph compilation. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '24). ACM, 4 2024. doi: 10.1145/3620665.3640366. URL https://pytorch.org/assets/pytorch2-2.pdf.

[3] C. Auer, M. Dolfi, A. Carvalho, C. B. Ramis, and P. W. Staar. Delivering document conversion as a cloud service with high throughput and responsiveness. In 2022 IEEE 15th International Conference on Cloud Computing (CLOUD), pages 363–373. IEEE, 2022.

[4] J. Berkenbilt. Qpdf: A content-preserving pdf document transformer, 2024. URL https://github.com/qpdf/qpdf.

[5] O. R. developers. Onnx runtime. https://onnxruntime.ai/, 2024. Version: 1.18.1.

[6] IBM. Data Prep Kit: a community project to democratize and accelerate unstructured data preparation for LLM app developers, 2024. URL https://github.com/IBM/data-prep-kit.

[7] A. S. Inc. PyMuPDF, 2024. URL https://github.com/pymupdf/PyMuPDF.

[8] J. Liu. LlamaIndex, 11 2022. URL https://github.com/jerryjliu/llama_index.

[9] M. Lysak, A. Nassar, N. Livathinos, C. Auer, and P. Staar. Optimized Table Tokenization for Table Structure Recognition. In Document Analysis and Recognition - ICDAR 2023: 17th International Conference, San José, CA, USA, August 21–26, 2023, Proceedings, Part II, pages 37–50, Berlin, Heidelberg, Aug. 2023. Springer-Verlag. ISBN 978-3-031-41678-1. doi: 10.1007/978-3-031-41679-8_3. URL https://doi.org/10.1007/978-3-031-41679-8_3.

[10] L. Mishra, S. Dhibi, Y. Kim, C. Berrospi Ramis, S. Gupta, M. Dolfi, and P. Staar. Statements: Universal information extraction from tables with large language models for ESG KPIs. In D. Stammbach, J. Ni, T. Schimanski, K. Dutia, A. Singh, J. Bingler, C. Christiaen, N. Kushwaha, V. Muccione, S. A. Vaghefi, and M. Leippold, editors, Proceedings of the 1st Workshop on Natural Language Processing Meets Climate Change (ClimateNLP 2024), pages 193–214, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics. URL https://aclanthology.org/2024.climatenlp-1.15.

[11] L. Morin, V. Weber, G. I. Meijer, F. Yu, and P. W. J. Staar. Patcid: an open-access dataset of chemical structures in patent documents. Nature Communications, 15(1):6532, August 2024. ISSN 2041-1723. doi: 10.1038/s41467-024-50779-y. URL https://doi.org/10.1038/s41467-024-50779-y.

[12] A. Nassar, N. Livathinos, M. Lysak, and P. Staar. Tableformer: Table structure understanding with transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4614–4623, 2022.

[13] B. Pfitzmann, C. Auer, M. Dolfi, A. S. Nassar, and P. Staar. Doclaynet: a large human-annotated dataset for document-layout segmentation. pages 3743–3751, 2022.

[14] pypdf Maintainers. pypdf: A Pure-Python PDF Library, 2024. URL https://github.com/py-pdf/pypdf.

[15] P. Team. PyPDFium2: Python bindings for PDFium, 2024. URL https://github.com/pypdfium2-team/pypdfium2.

[16] Y. Zhao, W. Lv, S. Xu, J. Wei, G. Wang, Q. Dang, Y. Liu, and J. Chen. Detrs beat yolos on real-time object detection, 2023.

6

## Appendix

In this section, we illustrate a few examples of Docling's output in Markdown and JSON.

arXiv:2206.01062v1 [cs.CV] 2 Jun 2022

# DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis

Bengt Pfitzmann

800 Research

Eurodellion, Switzerland

Christoph Auer

800 Research

Eurodellion, Switzerland

Michele Dolli

800 Research

Eurodellion, Switzerland

Ahmed S. Nasser

800 Research

Eurodellion, Switzerland

Peter Staer

800 Research

Eurodellion, Switzerland

## ABSTRACT

An open document-based analysis is a key requirement for high-quality PDF document conversion. With the recent availability of public large groups with datasets such as DoclayNet and DocMark, deep learning models have proven to be very effective at layout detection and segmentation. While these datasets are of adequate use to work with models they currently lack an longer available, many files are unrecorded as an example of the most important and well-informed, which is not possible. The most important and well-informed, which is not possible. The most important and well-informed, which is not possible. The most important and well-informed, which is not possible. The most important and well-informed, which is not possible. The most important and well-informed, which is not possible. The most important and well-informed, which is not possible. The most important and well-informed, which is not possible. The most important and well-informed, which is not possible.

## CCS CONCEPTS

1. Information | Information | Document structure | Applied computing | Document analysis | Computing and Technologies | Machine learning | Computer vision | Object detection

Copyright © 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2088, 2089, 2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 2100

![img-2.jpeg](img-2.jpeg)

Figure 4. Post-comparison of complex page layouts across the format document categories

## KEYWORDS

PDF document conversion, layout segmentation, object detection, data set, Machine Learning

Burg Pfitzmann, Christoph Auer, Michele Dolli, Ahmed S. Nasser, and Peter Staer. 2022. Doc LayNet: A Large Human Annotated Dataset for Document Layout Analysis. In Proceedings of the 2006 ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '22), August 14-18, 2022, Washington, DC, USA. ACM, New York, NY, USA. 5 pages. https://doi.org/10.1145/1000678.8000000

# DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis

Burg Pfitzmann 800 Research Foundation, Switzerland bpf.fzau@lcm.com

Christoph Auer 800 Research Foundation, Switzerland cawflyu@lcm.com

Michele Doh 800 Research Foundation, Switzerland dsof@uocn.lcm.com

Ahmed S. Nasser 800 Research Foundation, Switzerland ahn@uocn.lcm.com

Peter Staer 800 Research Foundation, Switzerland bse@uocn.lcm.com

## ABSTRACT

An open document-based analysis is a key requirement for high-quality PDF document conversion. With the recent availability of public large groups with datasets such as DoclayNet and DocMark, deep learning models have proven to be very effective at layout detection and segmentation. While these datasets are of adequate use to work with models they currently lack an longer available, many files are unrecorded as an example of the most important and well-informed, which is not possible. The most important and well-informed, which is not possible. The most important and well-informed, which is not possible. The most important and well-informed, which is not possible. The most important and well-informed, which is not possible. The most important and well-informed, which is not possible. The most important and well-informed, which is not possible. The most important and well-informed, which is not possible. The most important and well-informed, which is not possible.

## CCS CONCEPTS

1. Data layout analysis. — Document structure. — Applied computing. — Document analysis. — Computing methodologies. — Machine learning. — Computer vision. — Object detection

The results to mean digital or hard copies of part or of the work for conversion responses can be provided without the provided free copies and not made or distributed by credit or commercial advantage and that copies bear the costs and the full choice on the first page. Copyrights for third-party components of this work must be honored. For all other uses, contact the owner/partner(s).

KDD 10, August 14-18, 2022, Washington, DC, USA 5 (2022 Copyright held by the owner/partner), ACM ISBN 978-1-4000-8985-8000000. https://doi.org/10.1140/8000000000000000

Figure 1. Post-comparison of complex page layouts across different document categories

## KEYWORDS

PDF document conversion, layout segmentation, object detection, data set, Machine Learning

## ACM Reference Format:

Burg Pfitzmann, Christoph Auer, Michele Dolli, Ahmed S. Nasser, and Peter Staer. 2022. Doc LayNet: A Large Human Annotated Dataset for Document Layout Analysis. In Proceedings of the 2006 ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '22), August 14-18, 2022, Washington, DC, USA. ACM, New York, NY, USA. 5 pages. https://doi.org/10.1145/1000678.8000000

Figure 2: Title page of the DocLayNet paper (arxiv.org/pdf/2206.01062) - left PDF, right rendered Markdown. If recognized, metadata such as authors are appearing first under the title. Text content inside figures is currently dropped, the caption is retained and linked to the figure in the JSON representation (not shown).

7

KDD: 21, August 10–18, 2022, Washington, DC, USA. Hugh P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P. P.

Table 1: Prediction performance (mAP@0.5-0.01) of object detection networks on DocLayNet test set. The MRCNN (Bank R-CNN) and FRCNN (Foster R-CNN) models with ResNet-50 on ResNet-101 backbone were trained based on the network architecture from the detected model (see Bank R-CNN R101-FPN 5a, Foster R-CNN R101-FPN 5a), with default configurations. The VOLO implementation utilized was VOLOval [15]. All models were initialized using pre-trained weights from the COCO [47] dataset.

|   | Instruments | MRCNN |   | FRCNN |   | WALO  |
| --- | --- | --- | --- | --- | --- | --- |
|   |   |  R01 | R02 | R01 | R02  |   |
|  Cypress | 96.69 | 68.6 | 71.1 | 79.1 | 77.7 | 77.7  |
|  Protein | 95.95 | 70.9 | 71.8 | 75.7 | 77.2 | 77.2  |
|  Formula | 95.95 | 69.2 | 65.8 | 63.5 | 66.2 | 66.2  |
|  Last dose | 97.00 | 81.2 | 84.0 | 81.8 | 86.2 | 86.2  |
|  Page-Excess | 95.94 | 61.6 | 55.3 | 78.9 | 61.1 | 61.1  |
|  Page-Reality | 95.95 | 72.5 | 73.6 | 72.6 | 67.9 | 67.9  |
|  Perce | 89.92 | 74.7 | 72.5 | 72.6 | 77.1 | 77.1  |
|  Section branch | 89.94 | 67.6 | 67.5 | 68.8 | 78.6 | 78.6  |
|  Table | 97.91 | 82.2 | 82.5 | 82.2 | 86.3 | 86.3  |
|  Text | 96.96 | 86.6 | 84.8 | 85.4 | 88.1 | 88.1  |
|  Text | 96.92 | 76.7 | 80.4 | 79.9 | 82.7 | 82.7  |
|  All | 85.95 | 72.4 | 73.5 | 73.4 | 76.5 | 76.5  |

to avoid that at any cost in order to have clear, unbiased baseline numbers for human documents-based annotation. Third, we introduced the feature of mapping boxes around text segments to obtain a good accurate annotation and again rather time and effort. The 1/3 annotation is automatically divided every one shown loss to the minimum boundary loss around the enclosed text only for all yearly text-based segments, which includes only Table and Factor. For the latter, we unnoted annotation itself is an amount of understanding techniques while including all graphical lines. A downside of mapping boxes to enclosed text only is that some strongly passed PDF pages cannot be annotated correctly and merely be shipped. Fourth, we established a way to the pages to extend the case where an valid annotation according to the label problems could be achieved. Example cases for the model to PDF pages that render incorrectly or contain known that are apparently to capture with an overlapping, overlapping, high trained paper can not be annotated in the final dataset. With all these measures in place, experienced annotation staff managed to maintain a single page in a typical timeframe of this to win, depending on its complexity.

## 5 EXPERIMENTS

The primary goal of DocLayNet is to obtain high quality ML models capable of accurate document-based models on a wide variety of challenging layers. As discussed in Section 2, object detection models are currently the context to use, due to the standardization of ground-truth data in COCO format [16] and the availability of ground-truth data which is observed [17]. Prediction performance is made in the PacharNet and DocLainNet were obtained using matched object detection models such as Bank R-CNN and Foster R-CNN. As such, we will relate to these object detection methods in this

![img-3.jpeg](img-3.jpeg)

Figure 5: Prediction performance (mAP@0.5-0.01) of a Bank R-CNN network with ResNet-50 backbone trained on increasing fractions of the DocLayNet dataset. The learning curve (letions around the 0%) mark, indicating that increasing the size of the DocLayNet dataset with similar data will not yield significantly better predictions.

pages and lower the detailed evaluation of more recent methods mentioned in Section 2 for future work.

In this section, we will present several aspects related to the performance of object detection models in DocLayNet. Similarly, as in PolkayNet, we will evaluate the quality of their predictions using mean average precision (mAP) with 45 overlap that range from 0.5 to 0.70 in steps of 0.01 (mAP@0.5-0.01). These errors are computed by leveraging the evaluation code provided by the COCO [47] [16].

### Baselines for Object Detection

In Table 1, we present baseline experiments (given in mAP) on Bank R-CNN [12], Foster R-CNN [11], and VOLOv [15]. Both training and evaluation were performed in 8000 images with dimension as \(0.1\mathrm{m}^2\) of each plot. For training, we only used one annotation in case of substantially unnotated pages. As one can observe, the variation in mAP between the models is rather low, but overall between 0 and \(10\%\) lower than the mAP compared from the previous human annotations on high-annotated pages. This gives a good indication that the local-yetel dataset poses a worthwhile challenge for the research community to close the gap between human recognition and the approaches. It is interesting to see that Bank R-CNN and Foster R-CNN produce very comparable mAP scores, indicating that good based image segmentation derived from bounding boxes does not help to obtain better predictions. On the other hand, the more recent Vatoela model does very well and even our preformer humans on selected labels such as Text, Table and Fiction. This is not entirely surprising, as Text, Table and Fiction are abundant and the most visually distinctive in a document.

Table 2: Prediction performance (mAP@0.5-0.01) of object detection networks on DocLayNet test set. The MRCNN (Bank R-CNN) and FRCNN (Foster R-CNN) models with Predis(0.5) or ResNet-101 backbone were trained based on the network architecture from the detected model (see Bank R-CNN R101-FPN 5a, Foster R-CNN R101-FPN 5a), with default configurations. The VOLO implementation utilized was VOLOval [15]. All models were initialized using pre-trained weights from the COCO [47] dataset.

|   | Instruments | MRCNN | MRCNN | FRCNN | WALO  |
| --- | --- | --- | --- | --- | --- |
|   | 10,000 | 8,000 | 8,000 | 8,000 | 1,000  |
|  Cypress | 90.00 | 88.3 | 71.3 | 70.1 | 72.7  |
|  Formula | 90.00 | 76.9 | 71.8 | 70.7 | 71.6  |
|  Formula | 90.00 | 68.1 | 65.0 | 65.0 | 68.0  |
|  Lowest | 90.00 | 67.2 | 65.0 | 67.1 | 68.0  |
|  Page-Excess | 90.00 | 61.8 | 62.0 | 60.6 | 61.1  |
|  Page-Reality | 90.00 | 71.6 | 73.6 | 73.6 | 73.6  |
|  Perce | 90.01 | 71.7 | 73.7 | 73.6 | 73.1  |
|  Section branch | 90.00 | 67.5 | 65.0 | 65.0 | 71.0  |
|  Table | 77.00 | 62.2 | 62.0 | 62.0 | 62.0  |
|  Text | 90.00 | 64.6 | 62.0 | 62.0 | 68.1  |
|  Text | 90.02 | 76.7 | 80.4 | 79.6 | 82.7  |
|  All | 90.00 | 72.4 | 72.5 | 72.6 | 78.6  |

In detail that do not, due to a loss to have clear, unbiased baseline numbers for human document-based annotation. Third, we introduced the features of mapping boxes around text segments to obtain a good accurate annotation and again rather time and effort. The COO annotation was automatically divided every one shown loss to the relevant recording box around the enclosed text with the same number of words and other segments, which includes only Table and Fiction. The first, we introduced annotation staff to determine the number of corresponding information with the number of corresponding words. In the same, the corresponding words are not only one of the words, but they are not only one of the words, but they are not only one of the words, but they are not only one of the words, but they are not only one of the words, but they are not only one of the words, but they are not only one of the words, but they are not only one of the words, but they are not only one of the words, but they are not only one of the words, but they are not only one of the words, but they are not

## 5 EXPERIMENTS

The primary goal of DoclayNet is to obtain high quality ML models capable of accurate document-based models on a wide variety of challenging layers. As discussed in Section 2, object detection models are currently the market to use, due to the standardization of ground-truth data in COCO format [15] and the availability of ground-truth data in COCO format [15] and the availability of ground-truth data in COCO format [15]. Furthermore, machine-condition of PollayNet and the data were obtained using standard object detection models such as Bank R-CNN and Foster R-CNN. As such, we will relate to these object detection methods in this.

Figure 5: Prediction performance (mAP@0.5-0.01) of a Bank R-CNN network with ResNet-50 backbone trained on increasing fractions of the DocLayNet dataset. The learning curve follows around the 0% mark, indicating that increasing the size of the DocLayNet dataset with similar data will not yield significantly better predictions.

Japan estimated the detailed evaluation of more recent methods mentioned in Section 2 for future work.

In this section, we will present several aspects related to the performance of object detection models on DocLayNet. Similarly, as in PollayNet, we will evaluate the quality of their predictions using mean average precision (mAP) with 45 overlap that range from 0.5 to 0.70 in steps of 0.01 (mAP@0.5-0.01). These errors are computed by leveraging the evaluation code provided by the COCO [47] [16].

### Baselines for Object Detection

In Table 1, we present baseline experiments (given in mAP) on Bank R-CNN [12], Foster R-CNN [11], and VOLOval [15]. Both training and evaluation data performed in WES images with dimension of 0.01 + 0.01 pixels. For training, we only used one annotation in case of unbiasedly annotated pages. As we are shown, the number of mAP between the models is rather low, but overall between 0 and 0.01. Since then the mAP compared from the previous human annotations on high-annotated pages. The given system indicated that the DocLayNet dataset poses a worthwhile challenge for the relevant community to close the gap between human recognition and the approaches. It is interesting to see that Bank R-CNN and Foster R-CNN perform very comparable mAP scores, indicating that good based image segmentation derived from bounding boxes does not help to obtain better predictions. On the other hand, the more recent Vatoela model does very well and even our preformer humans on selected labels such as Text, Table and Fiction. This is something something, as Text, Table and Fiction are abundant and the most visually distinctive in a document.

Figure 3: Page 6 of the DocLayNet paper. If recognized, metadata such as authors are appearing first under the title. Elements recognized as page headers or footers are suppressed in Markdown to deliver uninterrupted content in reading order. Tables are inserted in reading order. The paragraph in "5. Experiments" wrapping over the column end is broken up in two and interrupted by the table.

8

A

|  class label | Count | % of Total |   |   | triple inter-annotator mAP @ 0.5-0.95 (%)  |   |   |   |   |   |   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|   |   |  Trans | Test | Val | All | Fin | Man | Int | Low | Pat | Tex  |
|  Caption | 22524 | 2.64 | 1.77 | 2.32 | 84.69 | 80.41 | 86.92 | 94.99 | 93.95 | 69.29 | n/a  |
|  Footnote | 6318 | 0.68 | 0.51 | 0.58 | 83.93 | n/a | 100 | 62.88 | 85.94 | n/a | 82.97  |
|  Formula | 25027 | 2.25 | 1.96 | 2.96 | 83.85 | n/a | n/a | 84.87 | 86.96 | n/a | n/a  |
|  Last-time | 185668 | 17.39 | 13.34 | 15.82 | 87.88 | 74.83 | 90.92 | 97.97 | 81.65 | 75.88 | 93.95  |
|  Page-factor | 70678 | 6.51 | 5.58 | 6.00 | 85.94 | 88.80 | 85.96 | 100 | 82.97 | 100 | 86.98  |
|  Page-header | 58022 | 5.38 | 4.78 | 5.06 | 85.89 | 68.76 | 90.94 | 98.100 | 91.92 | 97.99 | 81.88  |
|  Picture | 45976 | 4.21 | 2.78 | 3.31 | 69.71 | 56.59 | 82.88 | 69.82 | 80.95 | 66.71 | 59.74  |
|  Section header | 142884 | 12.68 | 13.77 | 12.85 | 83.84 | 76.81 | 90.92 | 94.95 | 87.94 | 69.73 | 78.88  |
|  Table | 34733 | 3.28 | 2.27 | 3.60 | 77.61 | 75.80 | 83.86 | 98.99 | 58.88 | 79.84 | 70.85  |
|  Text | 116577 | 10.82 | 90.28 | 85.00 | 84.86 | 81.86 | 88.93 | 88.93 | 87.92 | 73.79 | 87.95  |
|  Title | 5471 | 0.47 | 0.38 | 0.50 | 40.72 | 24.63 | 50.63 | 98.100 | 82.96 | 68.79 | 24.54  |
|  Total | 1107476 | 941123 | 99016 | 64.51 | 82.83 | 73.74 | 79.82 | 89.94 | 86.93 | 73.76 | 68.82  |

C

![img-4.jpeg](img-4.jpeg)

B

|   |  | % of Total | % of Total | % of Total | triple inter-annotator mAP @ 0.5-0.95 (%) | triple inter-annotator mAP @ 0.5-0.95 (%) | triple inter-annotator mAP @ 0.5-0.95 (%) | triple inter-annotator mAP @ 0.5-0.95 (%) | triple inter-annotator mAP @ 0.5-0.95 (%) | triple inter-annotator mAP @ 0.5-0.95 (%) | triple inter-annotator mAP @ 0.5-0.95 (%)  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  Start Date | Count | Total | Test | Val | All | Fin | Man | Int | Low | Pat | Tex  |
|  Caption | 28024 | 0.64 | 1.77 | 0.28 | 84.69 | 82.61 | 88.92 | 94.99 | 93.95 | 69.78 | n/a  |
|  Formula | 2018 | 0.63 | 0.51 | 0.58 | 83.24 | n/a | 100 | 62.88 | 85.94 | n/a | 82.97  |
|  Formula | 28027 | 0.63 | 1.96 | 0.28 | 83.85 | n/a | n/a | 64.87 | 86.96 | n/a | n/a  |
|  Last-time | 186888 | 17.19 | 13.44 | 15.82 | 87.88 | 74.83 | 90.92 | 97.97 | 81.65 | 75.88 | 83.95  |
|  Page factor | 70678 | 6.51 | 5.58 | 6.00 | 83.94 | 68.83 | 85.95 | 100 | 82.97 | 100 | 86.98  |
|  Page header | 28022 | 0.72 | 0.75 | 0.28 | 86.69 | 68.76 | 89.94 | 98.100 | 91.92 | 67.88 | 81.88  |
|  Picture | 45976 | 4.21 | 2.78 | 3.31 | 69.71 | 68.83 | 88.86 | 89.82 | 86.96 | 68.71 | 68.35  |
|  Section header | 142884 | 12.83 | 13.77 | 12.85 | 83.84 | 75.81 | 88.92 | 94.95 | 87.94 | 69.73 | 78.88  |
|  Table | 34733 | 0.63 | 0.57 | 0.60 | 77.61 | 75.80 | 88.86 | 88.86 | 88.83 | 73.84 | 78.88  |
|  Text | 116577 | 10.82 | 10.28 | 10.00 | 83.84 | 81.86 | 88.93 | 88.93 | 87.92 | 73.79 | 87.95  |
|  Title | 5471 | 0.47 | 0.38 | 0.50 | 83.72 | 24.83 | 88.93 | 94.100 | 82.96 | 69.79 | 24.85  |
|  Total | 1107476 | 941123 | 99016 | 64.51 | 82.83 | 73.74 | 79.82 | 89.84 | 86.93 | 73.76 | 88.85  |

Figure 4: Table 1 from the DocLayNet paper in the original PDF (A), as rendered Markdown (B) and in JSON representation (C). Spanning table cells, such as the multi-column header "triple inter-annotator mAP@0.5-0.95 (%)", is repeated for each column in the Markdown representation (B), which guarantees that every data point can be traced back to row and column headings only by its grid coordinates in the table. In the JSON representation, the span information is reflected in the fields of each table cell (C).

9