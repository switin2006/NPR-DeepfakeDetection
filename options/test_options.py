from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        
        # --- ADD THE MISSING '--resume' ARGUMENT ---
        parser.add_argument('--resume', type=str, required=True, help='path to the saved model checkpoint to test')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam (even if not used for training, BaseTrainer might expect it)')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam (BaseTrainer might expect it)')
        parser.add_argument('--optim', type=str, default='adam', help='optimizer to use [sgd, adam] (BaseTrainer might expect it)')
        parser.add_argument('--new_optim', action='store_true', help='new optimizer instead of loading the optim state (BaseTrainer might expect it)')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model (BaseTrainer might expect it)')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count (BaseTrainer might expect it)')
        parser.add_argument('--last_epoch', type=int, default=-1, help='starting epoch count for scheduler initialization (BaseTrainer might expect it)')
        # ==========================================================================================
        
        parser.add_argument('--model_path', type=str, default=None) # This might be redundant if --resume is used
        parser.add_argument('--no_resize', action='store_true')
        parser.add_argument('--no_crop', action='store_true')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')

        self.isTrain = False
        return parser
