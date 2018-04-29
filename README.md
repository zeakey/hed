## Reimplementation of [HED](https://github.com/s9xie/hed) based on official version of caffe

### For training:
1. Clone this code by `git clone https://github.com/zeakey/hed --recursive`, assume your source code directory is`$HED`;

2. Download [training data](http://vcl.ucsd.edu/hed/HED-BSDS.tar) from the [original](https://github.com/s9xie/hed) repo, and extract it to `$HED/data/`;

3. Build caffe with `bash $HED/build.sh`, this will copy reimplemented loss layer to caffe folder first;

4. Download [initial model](http://zhaok-data.oss-cn-shanghai.aliyuncs.com/caffe-model/vgg16convs.caffemodel) and put it
into `$HED/model/`;

5. Generate network prototxts by `python model/hed.py`;

6. Start to train with `cd $HED && python train.py --gpu GPU-ID 2>&1 | tee hed.log`.

### For testing:
1. Download [pretrained model](http://data.kaiz.xyz/edges/my_hed_pretrained_bsds.caffemodel) `$HED/snapshot/`;

2. Generate testing network prototxt by `python $HED/model/hed.py`(will generate training network prototxt as well); 

3. Run `cd $HED && python forward_all()`;

### Performance evaluation
I achieved ODS=0.779 on [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)
dataset, which is similar to HED's 0.78. Your can train your own model and evaluate using this
[code](https://github.com/zeakey/edgeval).

### Pretrained models and detection results:
| [Orig-HED](https://github.com/s9xie/hed)  | [My-HED](https://github.com/zeakey/hed) |
| ------------- | ------------- |
| [Pretrained model](http://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel)  | [Pretrained model](http://data.kaiz.xyz/edges/my_hed_pretrained_bsds.caffemodel)  |
| [BSDS results](http://data.kaiz.xyz/edges/detection_results/hed_pretrained_bsds.tar)  | [BSDS results](http://data.kaiz.xyz/edges/detection_results/my_hed_pretrained_bsds.tar)  |
| [Evaluation results](http://vcl.ucsd.edu/hed/eval_results.tar)  | [Evaluation results](http://data.kaiz.xyz/edges/my_hed_pretrained_bsds-eval.tar)  |

All detection results on the BSDS500 testing set and the pretrained models  are provided.
For example, the detected results of '3063.jpg' by the original [HED](https://github.com/s9xie/hed) and my
implementation are shown below:

<http://data.kaiz.xyz/edges/detection_results/hed_pretrained_bsds/3063.png>

![](http://data.kaiz.xyz/edges/detection_results/hed_pretrained_bsds/3063.png?x-oss-process=image/auto-orient,1/resize,h_250)

<http://data.kaiz.xyz/edges/detection_results/my_hed_bsds/3063.png>

![](http://data.kaiz.xyz/edges/detection_results/my_hed_bsds/3063.png?x-oss-process=image/auto-orient,1/resize,h_250)

You can preview results of all other images by replacing the filename in the above url.
___
By [KAI ZHAO](http://kaiz.xyz)

