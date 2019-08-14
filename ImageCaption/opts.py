import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='',
                    help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    # Data input settings
    parser.add_argument('--dataset', type=str, default='coco',
                    help='coco or pns')
    parser.add_argument('--input_json', type=str, default='data/coco.json',
                    help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_label_h5', type=str, default='data/coco_label.h5',
                    help='path to the h5file containing the preprocessed label')
    parser.add_argument('--input_topic_h5', type=str, default='data/coco/coco_gt_topics.h5',
                    help='path to the h5file containing the preprocessed image')
    parser.add_argument('--input_fc_dir', type=str, default='data/coco/feat299_resnet101_fc',
                    help='path to the fc feat fold')
    parser.add_argument('--input_att_dir', type=str, default='data/coco/feat299_resnet101_att',
                    help='path ot the attention feat fold')
    parser.add_argument('--use_img', type=int, default=0, 
                    help='whether to load raw image in dataloader (0/1 for no/yes)')
    parser.add_argument('--img_fold', type=str, default='data/cocc/images',
                    help='raw image fold path')
    parser.add_argument('--img_size', type=int, default=256,
                    help='the target size raw image will be resized')
    parser.add_argument('--img_csize', type=int, default=224,
                    help='the images are croped after resized')
    parser.add_argument('--cnn_model', type=str, default='resnet101',
                    help='cnn model for encoder resnet50/resnet101/resnet152/sceneprint are supported')
    parser.add_argument('--cnn_weight', type=str, default='resnet101.pth',
                    help='path to CNN pretrained weights')
    parser.add_argument('--start_from', type=str, default=None,
                    help="continue training from saved model at this path.")
    parser.add_argument('--start_from_best', type=int, default=0,
                    help="whether to start from the checkpoint with best score (0/1 for no/yes)")
    parser.add_argument('--old_id', type=str, default=None,
                    help="the id identity of the session to start off")
    parser.add_argument('--cached_tokens', type=str, default='coco-train-idxs',
                    help='Cached token file for calculating cider score during self critical training.')


    # Model settings
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--caption_model', type=str, default="show_tell",
                    help='show_tell, show_attend_tell, all_img, fc, att2in, att2in2, adaatt, adaattmo, topdown')
    parser.add_argument('--rnn_size', type=int, default=512,
                    help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in the RNN')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                    help='rnn, gru, or lstm')
    parser.add_argument('--input_encoding_size', type=int, default=512,
                    help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--att_hid_size', type=int, default=512,
                    help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')
    parser.add_argument('--fc_feat_size', type=int, default=2048,
                    help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--att_feat_size', type=int, default=2048,
                    help='2048 for resnet, 512 for vgg')
    parser.add_argument('--topic_num', type=int, default=200, 
                    help='topic number')
    parser.add_argument('--use_topic', type=int, default=0,
                    help='whether to use topic (0/1 for no/yes)')
    parser.add_argument('--use_fc', type=int, default=0,
                    help='whether to use the fully-connected layer output feat')
    parser.add_argument('--max_entity_length', type=int, default=6,
                    help='***not used***')
    parser.add_argument('--conv_channels', type=int, default=512,
                    help='***not used***')
    parser.add_argument('--conv_kernel_size', type=int, default=2,
                    help='***not used***')
    # Optimization: General
    parser.add_argument('--gpu_num', type=int, default=1,
                    help='how many gpus are used')
    parser.add_argument('--max_epochs', type=int, default=-1,
                    help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                    help='minibatch size')
    parser.add_argument('--iter_times', type=int, default=1,
                    help='average gradent over some iterations and got an update')
    parser.add_argument('--sample_weights', type=str, 
                    help='***not used***')
    parser.add_argument('--grad_clip', type=float, default=0.1, #5.,
                    help='clip gradients at this value')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--finetune_cnn_after', type=int, default=-1,
                    help='After what epoch do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
    parser.add_argument('--self_critical_after', type=int, default=-1,
                    help='After what epoch do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
    parser.add_argument('--seq_per_img', type=int, default=5,
                    help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
    parser.add_argument('--beam_size', type=int, default=1,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
    parser.add_argument('--cut_seq_p', type=float, default=-1,
                    help='***not used***')
    parser.add_argument('--cut_seq_p_start', type=int, default=0,
                    help='***not used***')
    parser.add_argument('--cut_seq_p_incre_rate', type=float, default=0.2,
                    help='***not used***')
    parser.add_argument('--cut_seq_p_incre_every', type=int, default=5,
                    help='***not used***')
    #Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam',
                    help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                    help='learning rate')
    parser.add_argument('--gate_decay_rate', type=float, default=0.0, 
                    help='***not used***')
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1,
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=3,
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8,
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--optim_alpha', type=float, default=0.8,
                    help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                    help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                    help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--momentum', type=float, default=0.9,
                    help='for SGD optim')
    parser.add_argument('--fix_rnn', type=int, default=0,
                    help='whether to fix rnn during training (0/1 for no/yes)')
    parser.add_argument('--hard_thresh', type=float, default=0.0,
                    help='***not used***')
    #Optimization: for the CNN
    parser.add_argument('--cnn_optim', type=str, default='adam',
                    help='optimization to use for CNN')
    parser.add_argument('--cnn_optim_alpha', type=float, default=0.8,
                    help='alpha for momentum of CNN')
    parser.add_argument('--cnn_optim_beta', type=float, default=0.999,
                    help='beta for momentum of CNN')
    parser.add_argument('--cnn_learning_rate', type=float, default=1e-5,
                    help='learning rate for the CNN')
    parser.add_argument('--cnn_weight_decay', type=float, default=0,
                    help='L2 weight decay just for the CNN')

    parser.add_argument('--scheduled_sampling_start', type=int, default=-1,
                    help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5,
                    help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05,
                    help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25,
                    help='Maximum scheduled sampling prob.')


    # Evaluation/Checkpointing
    parser.add_argument('--test_split', type=str, default='val',
                    help='split used to eval the performance val/test')
    parser.add_argument('--val_images_use', type=int, default=5000,
                    help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--dataset_ix_start', type=int, default=-1,
                    help='set the boundary to use only the subset of the wholed dataset. left boundary')
    parser.add_argument('--dataset_ix_end', type=int, default=-1,
                    help='set the boundary to use only the subset of the wholed dataset. right boundary')
    parser.add_argument('--save_checkpoint_every', type=int, default=2500,
                    help='how often to save a model checkpoint (in iterations)?')
    parser.add_argument('--checkpoint_path', type=str, default='save',
                    help='directory to store checkpointed models')
    parser.add_argument('--save_every', type=int, default=0,
                    help='save the checkpoint every x epochs')
    parser.add_argument('--language_eval', type=int, default=0,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--losses_log_every', type=int, default=25,
                    help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')

    # misc

    args = parser.parse_args()

    # Check if args are valid
    assert args.rnn_size > 0, "rnn_size should be greater than 0"
    assert args.num_layers > 0, "num_layers should be greater than 0"
    assert args.input_encoding_size > 0, "input_encoding_size should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.seq_per_img > 0, "seq_per_img should be greater than 0"
    assert args.beam_size > 0, "beam_size should be greater than 0"
    assert args.save_checkpoint_every > 0, "save_checkpoint_every should be greater than 0"
    assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
    assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
    assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"

    return args
