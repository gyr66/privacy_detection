# **非结构化商业文本信息中隐私信息识别**

## 问题背景

比赛赛题地址：[非结构化商业文本信息中隐私信息识别](https://www.datafountain.cn/competitions/472)

随着社交网络、移动通讯等技术的迅速发展，网络中存在大量包含隐私数据的文本信息，如何在非结构化的本文信息中精准识别隐私数据并对其进行保护已经成为隐私保护领域中亟需解决的问题。

本赛题将关注点集中在**隐私属性的识别问题**中，针对非结构化的本文信息进行分析，对文本中所涉及到的隐私信息精准提取。该任务为后续隐私保护操作提供强有力的支撑，是隐私保护领域的重要前提。

## 解决思路

这属于NLP领域的**传统实体识别任务（NER）**，限定实体类别，封闭文本语料。

常用的解决技术包括：

**规则方法**：使用正则表达式匹配邮箱、手机号等。

**序列标注**：对于每个token打标签，标签由实体位置信息（BIO）和实体类型信息（如PER、LOC、ORG）组成。通过这些标签可以识别出实体。

![image-20240104110736732](./picture/image-20240104110736732.png)

**序列生成：**

- **Seq2Seq**：输入文本作为源文本，生成标注后的目标文本。

- **GPT**：输入文本作为prompt，模型在prompt的基础上，生成标注后的目标文本。

（标注后的目标文本可以是方便抽取实体的任意格式，比如 *<佘达>[name];<建行>[company];<中弘北京像素>[company];<销售总监>[position];<龙坤>[name]* 。）

**阅读理解**：将实体识别转换为阅读理解任务，将输入文本作为context，对于要识别的不同实体类型可以设置不同的问题，比如：找出出现的人名，找出出现的地名等。（https://arxiv.org/abs/1910.11476）

**span枚举**：枚举所有可能的span，span通过分类器，判断span是否是某类实体。（https://aclanthology.org/D18-1309）

我们尝试了 **基于BERT的序列标注方法（NLU）** 和 **基于GPT的序列生成方法（NLG）**，在模型的预测输出之上，我们再利用 **正则表达式（规则方法）** 对手机号、微信号、QQ号进行匹配，提升在这三类实体上的召回率（Recall）。

## 数据集

我们首先对比赛方提供的数据进行分析：

1. 比赛方提供了训练集和测试集，其中：

   训练集样本数量：2514

   测试集样本数量：3955

2. 训练集样本类别：

   隐私数据类别有如下14类：position、name、movie、organization、company、book、address、scene、mobile、email、game、government、QQ、vx。

对比NER经典数据集conll2003（训练集数据有14k，只有PER、ORG、LOC、MISC四类），可以看出本比赛数据集**数据量很少**，而且**实体种类很多**。如何**使用少量的数据对模型进行训练**，**减轻模型过拟合**，**保持模型的泛化能力**，是这个**比赛的关键点**所在。

## 序列标注

序列标注是解决NER问题的经典方法，常用的模型有：

- **CRF**

- **LSTM**（ https://arxiv.org/abs/1508.01991 ）

- **BERT**（ https://arxiv.org/abs/1810.04805 ）

我尝试了BERT、RoBERTa、ERNIE-3.0，带CRF头以及不带CRF头。

为了解决过拟合问题，我尝试使用**LoRA**（ https://arxiv.org/abs/2106.09685 ），减少可训练的参数数量。

### 数据预处理

为了做序列标注，首先需要对比赛方提供的数据进行预处理，转换成**BIO标注**的格式，流程如下：

1. 验证文本比赛方提供的数据中的Pos_b、Pos_e和Privacy字段是否对应，修正或移除不对应的数据；
2. 获取标签列表：对实体使用B、I标注，对非实体使用O标注；
3. 将文本切分成字符列表，并验证字符列表长度和标签列表长度相同；
4. 创建标签到索引的映射，将标签列表转换成对应索引的列表。

![image-20240104122330494](./picture/image-20240104122330494.png)

为了方便后面测试模型，我将预处理好的数据集使用Hugging Face的datasets库进行了封装，上传到了Hugging Face仓库中：https://huggingface.co/datasets/gyr66/privacy_detection 。

![image-20240104122353107](./picture/image-20240104122353107.png)

### BERT

我首先尝试使用BERT。使用BERT进行序列标注，只需要将最后一层各个token的hidden state过一下分类头即可。我使用Hugging Face的**transformers**加载预训练模型，**trainer**进行模型训练，**evaluate**进行模型评估。

**由于tokenize之后的token列表可能和输入字符列表长度不同**，这就导致token列表可能没有和label对齐。为了解决这个问题，可以使用Hugging Face的Fast Tokenizer，借助word_ids获得token和输入字符之间的映射，从而得到token和label之间的映射。对于添加的特殊字符（如\<CLS\>、\<SEP\>、\<PAD\>），无法找到对应的label，此时可以将label置为-100，PyTorch的Cross Entropy Loss在计算时会自动忽略这些位置的损失。

模型训练脚本请见： https://github.com/gyr66/privacy_detection/blob/master/train.py 。

数据集是中文，因此要选择中文的BERT，比如bert-base-chinese。为了更加充分发挥预训练模型的能力，我选择了在NER数据集上微调过的BERT作为预训练模型：https://huggingface.co/Danielwei0214/bert-base-chinese-finetuned-ner 。

在比赛数据集上微调，选择验证集（从比赛训练集上切分0.15做验证集）上F1最高的模型作为最终模型。模型在验证集上表现如下：

![image-20240104122557092](./picture/image-20240104122557092.png)

训练好的BERT模型地址：https://huggingface.co/gyr66/bert-base-chinese-finetuned-ner 。

### RoBERTa

RoBERTa在更大的语料上进行了预训练，使用了更大的批量大小，并且使用了动态掩码，在多项任务上表现比BERT要好。我选择了[hfl](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)[/](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)[chinese](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)[-](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)[roberta](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)[-](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)[wwm](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)[-](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)[ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)[-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)作为预训练模型，它在预训练阶段采用了**全词MASK（wwm）**（对组成同一个词的汉字全部进行Mask ）。

![image-20240104122709125](./picture/image-20240104122709125.png)

在比赛数据集上微调，选择验证集上F1最高的模型作为最终模型。模型在验证集上表现如下：

![image-20240104122717770](./picture/image-20240104122717770.png)

训练好的RoBERTa模型地址：https://huggingface.co/gyr66/RoBERTa-ext-large-chinese-finetuned-ner 。

### RoBERTa + CRF

使用BERT或者RoBERTa，在对token进行分类时，没有考虑token之间的关联。比如B-person后面不应该再跟B-person，而应该是I-person。然而使用BERT或者RoBERTa进行序列标注时有时会出现两个B相连的情况。

为了解决这个问题，可以加入一个CRF头。CRF会将整个标注序列一次解码出来（而不是一个token一个token的解码），考虑了不同标签之间转移的概率（比如B-X到B-X的概率应该很小）。

![image-20240104123129845](./picture/image-20240104123129845.png)

我使用pytorch-crf库（ https://github.com/kmkurn/pytorch-crf ），将其与Hugging Face的预训练模型进行整合，方法如下：

1. 创建模型类**BertCrfForTokenClassification**，继承**BertPreTrainedModel**；

   ![image-20240104123258230](./picture/image-20240104123258230.png)

2. 为模型添加**CRF头**和**分类头**，并加载基座模型的权重；

   ![image-20240104122926234](./picture/image-20240104122926234.png)

3. 重写forward方法，借助attention_mask生成CRF头需要的**labels_mask**，遮蔽住特殊token，使得CRF头仅关注原始序列。调用decode方法从logits中解码出序列，调用forward方法由logits、labels、mask计算loss（如果传递了label）；

   ![image-20240104122941950](./picture/image-20240104122941950.png)

4. 由于使用trainer进行训练，forward需要返回**TokenClassifierOutput**类型，以兼容trainer的训练流程。

   ![image-20240104123054282](./picture/image-20240104123054282.png)

模型代码请见：https://github.com/gyr66/privacy_detection/blob/master/model.py 。

在比赛数据集上微调，选择验证集上F1最高的模型作为最终模型。模型在验证集上表现如下：

![image-20240104123524430](./picture/image-20240104123524430.png)

可以看到采用这样的训练策略后，模型的F1值超过了不加CRF头的RoBERTa。

训练好的模型地址：https://huggingface.co/gyr66/RoBERTa-ext-large-crf-chinese-finetuned-ner-v2 。

可以看到，相比于不加CRF头，模型F1值还减少了（之前F1值是0.7318）。我推测可能是因为BERT已经学习到了token之间的关联，因此CRF头对模型效果提升不明显。由于引入了新的参数，而训练数据量很少，模型并没有被训练很好，导致F1值减少。

训练好的RoBERTa + CRF模型地址： https://huggingface.co/gyr66/RoBERTa-ext-large-crf-chinese-finetuned-ner 。

根据推测，模型效果不好可能是因为训练数据少导致CRF头参数没有被训练很好。于是我采取了以下策略：**采用分层学习率：CRF头的学习率设置为底座模型学习率的100倍**；在比赛数据集上微调，选择验证集上F1最高的模型作为最终模型。模型在验证集上表现如下：

 ![image-20240110143814590](./picture/image-20240110143814590.png)

为了方便使用自定义的模型BertCrfForTokenClassification，将模型注册到AutoModelForTokenClassification，并将模型定义文件上传到Hugging Face仓库中。这样可以直接使用AutoModelForTokenClassification.from_pretrained()方法加载使用训练好的带有CRF头的自定义模型，并且可以和token classification的pipeline无缝集成。

![image-20240104123622473](./picture/image-20240104123622473.png)

![image-20240104123628149](./picture/image-20240104123628149.png)

![image-20240104123632595](./picture/image-20240104123632595.png)

### RoBERTa + LoRA

在前面的实验中，我发现模型出现了严重的过拟合问题，从下图中可以看到**验证集损失约为训练集损失的700倍！**

![image-20240104123726865](./picture/image-20240104123726865.png)

我认为过拟合的主要原因有两个：

1. 模型太大，可训练的参数太多，很快就把训练集的数据全记住了，丢失了泛化能力；
2. 训练集数据太少。

训练集数据是比赛方给的，没法进行扩充。因此，解决过拟合的问题，主要策略是缩小模型规模。然而，大型预训练模型比小型模型包含更丰富的知识。为了能够利用大型模型的知识，同时减轻过拟合，我选择采用**LoRA**（Low-Rank Adaptation）（ https://arxiv.org/abs/2106.09685 ），冻结预训练模型的权重，而仅仅训练少部分额外权重，然后将训练的权重合并到预训练模型中。这种方式大幅度降低了可训练参数的数量，从而有望有效减缓过拟合的问题。

![image-20240104123747801](./picture/image-20240104123747801.png)

通过前面的实验，在验证集上最好的模型是不加CRF头的RoBERTa，因此我将此模型作为基座模型，训练LoRA参数。

借助Hugging Face的**peft**，可以很容易地使用LoRA对模型进行微调。

创建一个LoRA config，配置矩阵的秩等参数。然后使用get_peft_model方法包装模型即可。

![image-20240104123824420](./picture/image-20240104123824420.png)

获取模型可训练的参数数量：

*trainable params: 422,941 || all params: 324,925,498 || trainable%: 0.1301655310535217*

可以看到**可训练的参数数量只有总参数数量的0.13%**，这有希望减轻模型过拟合的问题！

在比赛数据集上微调，选择验证集上F1最高的模型作为最终模型。模型在验证集上表现如下：

![image-20240104123902214](./picture/image-20240104123902214.png)

可以看到，该方法并没有取得很好的效果。**验证集上的损失和训练集上的损失并没有之前相差如此巨大，验证集上的损失也要比RoBERTa的损失（0.7697）要小很多，但是F1值却比较小**。

![image-20240104123953035](./picture/image-20240104123953035.png)

训练好的LoRA参数地址： https://huggingface.co/gyr66/RoBERTa-ext-large-lora-chinese-finetuned-ner 。

尝试使用LoRA减少训练参数的数量来减轻模型过拟合并没有取得很好的效果。我意识到要想真正用少量训练数据得到效果好的模型，仅仅采用trick是没有用的。解决训练集数据量少问题的关键在于**引入外部知识（先验知识）**。比如模型应该在即使没有训练数据的情况下，也能识别哪些是人名、地名等。通过少量数据，向模型示例要找的实体是哪些（即告诉模型某一类实体的含义是什么，比如position，要找的是一些职位实体，如院长、教授等），这样只要基座模型先验知识足够多，应该能够取得很好的效果！为此我首先尝试了百度的**ERNIE 3.0**（ https://arxiv.org/abs/2107.02137 ）。

![image-20240110144037446](./picture/image-20240110144037446.png)

在比赛数据集上微调，选择验证集上F1最高的模型作为最终模型。模型在验证集上表现如下：

![image-20240110144052708](./picture/image-20240110144052708.png)

训练好的模型地址： https://huggingface.co/gyr66/Ernie-3.0-large-chinese-finetuned-ner 。

可以看到模型的效果还是不能让人满意。但是我认为引入先验知识的思路是对的，只是目前基座模型比较小，无法取得非常好的效果。于是我将目光转向**大语言模型**。大语言模型具有非常丰富的先验知识，而且有很强的泛化能力，做这项任务应该能够取得很好的效果。

**大语言模型的尝试请见下一节！**

### 最终效果

序列标注方法是解决NER问题的经典方法。我尝试了BERT、RoBERTa、ERNIE-3.0、CRF等模型。所有尝试过的模型都上传到了模型仓库中：https://huggingface.co/gyr66 ，方便后面做模型聚合。

![image-20240110144125805](./picture/image-20240110144125805.png)

最终，经过测试，基于ERNIE-3.0的模型：https://huggingface.co/gyr66/Ernie-3.0-large-chinese-finetuned-ner ，在比赛方的测试集上表现最好。

inference脚本请见：https://github.com/gyr66/privacy_detection/blob/master/inference.ipynb  。

提交评测，在**测试集上F1值约为0.72**。

![image-20240110144747108](./picture/image-20240110144747108.png)

![image-20240104124025627](./picture/image-20240104124025627.png)

## 序列生成

起初，我们考虑到比赛提供的训练集较小，为了在ZERO-SHOT测试中取得更好的表现，我们选择了微调LLM（Large Language Model）的方法。为了平衡参数量和原始性能，我们首先采用了ChatGLM3-6b-base作为训练生成模型。据称，ChatGLM3-6b在10b以下拥有最强性能，比较适合序列生成任务。

### 初始效果

在初始效果上，通过简单测试证明LLM生成分类答案是可行的。随后，我们创建了训练集并使用官方的微调代码进行了1000步的训练。然而，在测试集上的结果表现并不理想，GLM3-6b-base似乎无法完全理解输入文本的含义，经常不按照提示进行输出，生成的答案甚至与输入的大小写不一致。

**不按照格式输出**

<img width="622" alt="image" src="https://github.com/gyr66/privacy_detection/assets/83167730/609929f3-57d4-498c-a716-9ed26592281d">

**生成答案与输入大小写不同**

<img width="201" alt="image" src="https://github.com/gyr66/privacy_detection/assets/83167730/0c323312-cd3f-4872-8a66-f96c4953b6af">



### LoRA微调

为了解决这个问题，我们认为训练集的训练次数可能不足，因此我们决定增加训练步数到10000步（每100步存储一次LoRA权重）。然而，在加载了10000步的LoRA权重后进行测试，评分下降了2%。我们认为10000步的lora权重使得生成的response过于细致，同时还产生了大量错误答案，准确度降低最终取得更差的结果。
<img width="241" alt="image" src="https://github.com/gyr66/privacy_detection/assets/83167730/139d4cfe-8a62-4451-863d-9fc65256a8c8">


### 最终效果

最终，我们尝试了更改PROMPT、选择合适的LoRA权重，并调整了测试结果生成代码。在评测中，我们获得了0.621的得分，但这仍然远远不及在序列标注的结果。

<img width="572" alt="image" src="https://github.com/gyr66/privacy_detection/assets/83167730/542d7773-5e6d-41c8-894b-d5d28ac8ad80">


### 测试集文件生成

在生成predict.csv时，鉴于不同的测试文件可能导致代码中断，我们改变了策略，先生成response文件，然后再读取对应输入文本和回答文本，获得命名实体和种类。通过查找实体在原输入文本中的位置，我们生成了测试集文件。详情见make_predict.py

### ChatGLMm3-6b-chat

我们也测试ChatGLM-6b-chat的模型，以期望chat模型能好的按照输入的Prompt去生成答案。但在增加训练步数之后，base模型可能较好的实现格式化输出，且chat-6b模型加载相同权重表现不如base-6b模型，故没有在PPT中进行chat-6b相关的实验内容。

### 总结

总体而言，使用LLM完成分类任务是可行的。

LLM经过大量文本预训练后具有一定的泛化性，但在具体的分类问题中仍需要经过微调。

在生成文本方面，Decoder存在一些问题，LM更依赖自身的预训练来生成答案，这使得生成的实体和原输入文本大小写不同，同时ChatGLM-6b对繁体字的识别分类能力不足。

LLM的参数量较大，更依赖大量的训练数据，同时其内部性能尚不能有效地引导运用和解释，经过LoRA微调后的性能受到一定限制。这也使得其效果不如RoBERTa等较“小”的模型，。

## 规则法 + Ensemble

### 正则表达式

​	在传统的命名实体识别（NER）任务中，使用正则表达式的规则方法也是一种常用手段。正则表达式的使用方法通常有两种。其一，基于正则表达式的匹配帮助查找或匹配文本中的特定模式。例如，“张晓明老师”、“王小二同学”，并预定义为人名实体。其二，使用正则表达式提取具有强规则性的实体。例如，电话号码、邮箱地址等等，这些实体有固定的格式或规则，可以直接通过正则表达式来提取。

​	然而，正则表达式在NER中作用有限。它只能作用于抽取像邮箱、电话、日期这类有固定格式的实体、或像人名、地名、机构名这类有一定模式的实体。当面临复杂的、不规则的、没有明确格式的实体时，比如情感表达、观点抽取等，单纯使用正则表达式就力不从心了。此外，受限于正则表达式的单一匹配模式，在处理多模式、模糊模式的实体时，也可能造成误匹配的情况。因此，在目前的NER任务中，正则表达式通常与基于统计学的方法（如隐马尔可夫模型-HMM、条件随机场-CRF等）一同作为辅助手段辅助深度学习方法完成实体识别。

![image-20240111134558353](picture/ensemble1.png)

### Ensemble

​	为进一步提高性能，我们引入了Ensemble模型。它的基本思想是结合一组模型的预测结果来产生最终的预测结果。一般情况下，组合起来的模型比任何一个单独的模型都要好。根据集成的方法，主要可以分为几种类型：Bagging、Boosting和Stacking。

* Bagging：通过并行训练一组独立的模型，然后共同投票或者平均预测结果。每个模型都是在数据集的不同随机子集上进行训练的，其中随机森林就是最著名的例子。
* Boosting：这种方法是以序列的方式训练模型，每个新的模型都根据前面模型的性能进行优化调整。其目的是减少总体错误率，比如AdaBoost及梯度提升就是此类案例。
* Stacking：在这种策略中，我们会训练一些模型，然后用一个新的模型将其输出组合起来，以得到最终结果。第二层的模型被用于学习如何最好地结合前面模型的预测。

​	目前，本项目主要使用Ensemble模型，未来将会探索更多的ensemble模型。我们从模型仓库：https://huggingface.co/gyr66 中选择以下六种模型进行聚合：

![ensemble2](picture/ensemble2.png)

​	其中的model1、model2、model3为lora微调模型需要使用peft读取base后进行合并，如下所示

![ensemble2](picture/ensemble3.png)

![ensemble2](picture/ensemble4.png)

![ensemble2](picture/ensemble5.png)

### 数据处理

​	 由于每各个模型对输入要求不同，将序列输入每个模型前按照其最大长度对文本进行切割：

![ensemble2](picture/ensemble6.png)

​	由于每个模型的输出不同，部分token没有预测值。因此将每个模型的预测结果按输入token的顺序放入一个新的字典列表，如果某个token没有被预测出结果，那么其对应的位置放入一个空字典，以保证投票的对齐

![ensemble2](picture/ensemble7.png)

### 聚合

​	使用对计数器对每个输入的预测值进行计数，选择数量最多的entity作为该位置的最终结果

![ensemble2](picture/ensemble8.png)

​	根据前后的entity进行链接得到最终的识别结果

![ensemble2](picture/ensemble9.png)

包含以上所有步骤的Ensemble脚本请见：https://github.com/gyr66/privacy_detection/blob/master/Ensemble.ipynb

## 总结

我们选择了非结构化商业文本信息中隐私信息识别这个比赛题目，从非结构化文本中识别出隐私类别实体。
比赛方提供的数据集总共有14类隐私实体，训练集样本数量2.5K。相比较于传统NER任务，实体类别大大增多，训练数据量较少，这是这个比赛的难点所在。
我们首先尝试了解决NER问题经典的序列标注方法，尝试了BERT、RoBERTa、RoBERTa + CRF、RoBERTa + LoRA、ERNIE-3.0等模型。总共训练了13个模型。为了提升比赛分数，我们使用了Ensemble方法，并引入了正则表达式，标注出模型没有识别的邮箱、手机号、QQ号等隐私实体。通过序列标注方法的尝试，我们发现由于训练集样本数量太少，模型过拟合严重。我们认为，解决这个问题的方法是引入先验知识，即模型应该在没有样本的情况下，也能识别出人名、地名这些比较简单类别的实体。而对于比较复杂的实体，比如position，scene等，通过少量样本示例模型，模型能够明白要找的实体类别是什么意思，从而能够很好地在句子中找到对应实体。
于是我们将目光转向大语言模型，大语言模型具有很多先验知识，很符合我们的需求。我们尝试了非常优秀先进的开源GLM模型，设计了模型的输出格式，使用训练集进行微调。首先尝试使用LoRA微调了base版本，发现模型有时不按照输出格式输出，指令遵循能力不强。于是我们又使用p-tuning微调了chat版本。这种方式最终并没有取得很满意的效果。我们认为可能是因为生成式模型并不擅长输出复杂的格式化信息，而且输出的内容可能原序列中没有，造成误差。我们现在认为更好地做法是基于阅读理解的方式，对于每个隐私实体类别训练一个识别模型，通过在指定实体类别上微调，模型应该能够很好地完成这类实体的识别任务。最后将所有模型的识别到的实体组合起来。这种方式可以避免让模型输出复杂结构化信息。未来我们将尝试这种方法，并将结果更新到代码仓库中（ https://github.com/gyr66/privacy_detection ）。
