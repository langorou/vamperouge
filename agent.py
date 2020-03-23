class Agent:
    def __init__(self, name, model):
        self.name = name
        self.model = model


		self.train_overall_loss = []
		self.train_value_loss = []
		self.train_policy_loss = []
		self.val_overall_loss = []
		self.val_value_loss = []
		self.val_policy_loss = []

    def simulate(self):
        # TODO
