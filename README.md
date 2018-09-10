# Remote sensing image retrieval
 
this is an implement of [DenseNet]() using keras ,this project can do Remote sensing image classifiy or retrieval.I trained and evaluated this model on a dataset called [PatternNet](https://www.researchgate.net/publication/317558235_PatternNet_A_Benchmark_Dataset_for_Performance_Evaluation_of_Remote_Sensing_Image_Retrieval).

## Dependencies

The project was tested in the following environment

- python 3.5.2
- Keras 2.1.6
- h5py
- tensorflow-gpu 1.9.0
- pillow(PIL) 5.0.0
- numpy 1.14.0
- sklearn

I build this project in Windows 10 so there maybe some path problems when rebuild it in linux/MacOSX.

You may also need a GPU to speed up the train process,so  installtion of the CUDA/CUDNN kit maybe necessary.

 
## How to use

### Data setup

Using my PatternNet dataset or your own dataset,all you need to do is

+ Create a new folder,such as

>myDataset

+ for each catagory,Create a new folder in 'myDataset' 
>myDataset/
>>catagory1
>>catagory2
>>catagory3
>>...

+ put the image into these folders

After this,you need to run `utils.write_csv()`.This function need 3 parameters `image_folder, csv_train_path, csv_test_path`.This function automatically divides the data into training and test set,and the test set is 30 percent of the total dataset.

	import utils

	image_folder = r'PatternNet'
    csv_train_path = r'PatternNet_train.csv'
    csv_test_path = r'PatternNet_test.csv'
    	
    write_csv(image_folder, csv_train_path, csv_test_path)

### Models
To use pretrained weights,you should set the `weights='imagenet'` when reference `DenseNet.DenseNet()`

When training your own dataset from scratch,simply set the `weights=None`.

#### Train

To train a model use
	
	from train_and_val import train

	lr = 1e-6
    epochs = 70
    batch_size = 16
    classes = 38
    image_size = 224
    model_path = 'pattern.h5'
    log_filepath = 'pattern_log'
    csv_train_path = r'PatternNet_train.csv'
    csv_test_path = r'PatternNet_test.csv'

	train(image_size,classes,
          csv_train_path,csv_test_path,
          batch_size,epochs,lr,
          log_filepath,model_path)

The parameter above can be changed to adjust model performance.

After training,you can call `tensorboard` to see the results in detail.use `cd` command to navigate to the project folder,then

`>tensorboard --logdir log_filepath`

#### Image classification

To do classification job,simply use the weights file that you trained in the last step,and call `train_and_val.validate()`.Here are the demo:

	from train_and_val import validate
	
	image_size = 224
	model_path = 'pattern.h5'
	classes = 38
	csv_val_path = r'PatternNet_test.csv'
	csv_result_path = r'PatternNet_result.csv'

	validate(image_size,classes,
              model_path,csv_val_path,csv_result_path)

The result is saved `csv_result_path`.

#### Image retrieval

This function contains two parts: `index` and `retrieval`

##### index

This is an implement of feature extraction.The DenseNet's avg pool layer is used to extract features .`img_path,label,features` are saved into index_file.
	
	from train_and_val import index

	image_size = 224
	model_path = 'pattern.h5'
	classes = 38
	csv_imageLib_path = r'PatternNet_test.csv'
	index_file = r'PatternNet_index.h5'

	index(image_size,classes,
          model_path,csv_imageLib_path,index_file)

##### retrieval

This function use Euclidean distance to find out the image which is similar as target image.Note that `target_path` need to be a csv file which formatted as `image_path,category`,only one image should be in this file
	
	from train_and_val import retrieval

	image_size = 224
	model_path = 'pattern.h5'
	classes = 38
	target_path = r'target.csv'
	retrieval_result_file = r'retrieval_result.txt'

	retrieval(image_size,classes,
				model_path,target_path,index_file,
				retrieval_result_file)

## Other things

This is my second project in Github,hope you can star or fork this project~

Any issue,you can contact me on QQ `2043494361` or just email me at `204349461@qq.com`.

Hope you enjoy this~