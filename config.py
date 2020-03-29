# Number of iterations when learning
num_iters = 100
# Number of complete self-play games to simulate during a new iteration
num_eps = 100
temperature_threshold = 30
# During arena playoff new neural net will be accepted if threshold or more of games are won
update_threshold = 0.55
# Number of game samples to train the neural networks
max_queue_length = 200000
# Number of games moves for MCTS to simulate
num_MCTS_sims = 25
# Number of games to play during arena play to determine if new net will be accepted
arena_compare = 40
cpuct = 1
checkpoint = "./temp/"
load_model = True
load_folder_file = ("temp", "checkpoint_0.pth.tar")
load_samples = True
load_samples_folder_file = ("temp", "checkpoint_0.pth.tar.samples")
skip_first_self_play = True
delete_old_samples = True
num_iters_for_train_samples_history = 4

board_width = 16
board_height = 16

nn_inplanes = 3
nn_planes = 256
nn_residual_layers = 4
nn_vh_hidden_layer_size = 256

train_lr = 0.002
train_epochs = 10
train_bs = 64
