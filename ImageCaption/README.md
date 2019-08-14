
## Code for Paper "Topic-Oriented Image Captioning Based on Order-Embedding"

### Requirements  
* Python2.7, java 1.8.0, PyTorch 0.4.1, theano

### Data process
* We use [karpathy's splits](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip).
* The processed captions and pretrained topics can be downloaded from [here](https://drive.google.com/file/d/1sVgQ76GzyC0GCps-cToSDL2udt3gSHuO/view?usp=sharing). Unzip it and put the folder "coco" in ImageCaption/data/.
* Topic model training process can be found in topic_model.ipynb

### Prepro image feats 
* Download the cnn checkpoint from [here](https://drive.google.com/file/d/1Ucm013BEHtEpzHlzo5IMxPB-gCTKKGsI/view?usp=sharing)
* Go to ImageCaption    
`>>> bash prepro_feats.sh`

### Order embedding training
* Download the processed dataset from [here](https://drive.google.com/file/d/1tp1caeLukSbET1ufaGwZr8Ry3w3Xb-C7/view?usp=sharing)    
* Go to order-embedding  
`>>> bash batch_jobs.sh`

### Image captioning training
* Training without topic    
`>>> bash trainWithoutCNN.sh`
* Train with topic   
`>>> bash trainWithoutCNN_t.sh`



















