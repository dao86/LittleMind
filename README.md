首先需要感谢MiniMind项目  
地址:https://github.com/jingyaogong/minimind  
MiniMind代码简练易懂，说明中提供的训练过程非常适合学习和实验。  
因此本项目从minimind中获得了很多帮助。    

如何训练:  
首先需要切换当前目录为项目路径  
cd xxx/xxx/xxx  
然后运行命令   
python LittleTrainer.py --mode pre  --data_path ./data/xxx.jsonl  --dim 512  --layernum 8  
参数说明:  
--mode pre ,进行pre训练  
 <img src="./img/pre_trainer.png" width="50%">   

--mode sft ,进行sft训练  
 <br/><img src="./img/sft_trainer.png" width="50%">   

--data_path ./data/xxx.jsonl ,数据集文件地址   
--dim 512 ,模型维度 默认512  
--layernum 8,模型层数 默认8  
数据集的获取:  
1、魔搭社区 数据集 搜索minimind   
注意，此项目目前还没提供tokenizer训练，只能使用现有的，bos,eos都使用的是qwen风格,看数据集时需要注意  

1、提供保存训练进度的sft支持checkpoint功能，但只支持pre sft训练，dpo和orpo目前还不支持    
--train_id xxxx (训练任务id，此参数功能必需)  
--num_checkpoint 10 (每个eopchs 的保存次数.如果设置为0的话，默认使用checkpoint_interval参数)
--checkpoint_interval 100（每n步进行一次 checkpoint  此参数功能必需）  

2、提供单机多卡功能，同样目前只支持pre,sft  
--ddp 1  (--ddp 0 不支持)  
--world_size 2 (gpu数量)  
其他参数可以参考代码LittleTrainer.py 中的 set_train_args方法 ,如epochs,bacth_size等

如何测试:  
可以直接运行python test.py  --mode pre --dim 512 --layernum 8     
--mode pre ,读取pre模式训练的model文件并进行推理  
--mode sft ,读取sft模式训练的model文件并进行推理  
--dim 512 ,模型维度,默认512,与训练时使用的参数要符合 
--layernum 8,模型层数 ,默认8,与训练时使用的参数要符合 

最后是训练测试效果，如下  

 <br/><img src="./img/loss_pre_512.png" width="50%"> 
 <br/><img src="./img/loss_sft.png" width="50%"> 
 <br/><img src="./img/test.png" width="50%"> 
