import flwr as fl
from typing import Optional, Tuple
from models import *
from pathlib import Path

import os
from flwr.common.parameter import ndarrays_to_parameters

def get_initial_parameters(net):
    """Returns initial parameters from a model.

    Args:
        num_classes (int, optional): Defines if using CIFAR10 or 100. Defaults to 10.

    Returns:
        Parameters: Parameters to be sent back to the server.
    """
    model = net
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = ndarrays_to_parameters(weights)

    return parameters




os.environ['GRPC_POLL_STRATEGY'] = 'epoll1'

# net_g = Generator(infc=256, nfc=128)
# netG_A2B = Generator_UGATIT(image_size=256)
net_d_style = Discriminator(nfc=128 * 3, norm_layer=nn.BatchNorm2d)
initial_parameters = get_initial_parameters(net_d_style)
### FedAvg/FedBN/FedAdam..

# strategy = fl.server.strategy.FedAvg(
#         # min_fit_clients=2,
#         # min_evaluate_clients=2,
#         min_available_clients=2,
#         # initial_parameters=initial_parameters,
#     )

## FedProx
strategy = fl.server.strategy.FedProx(
        min_fit_clients=1,
        min_evaluate_clients=7,
        min_available_clients=7,
        proximal_mu=1.0
    )

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=71),
    strategy=strategy,
    grpc_max_message_length=1500*1024*1024,
    )




