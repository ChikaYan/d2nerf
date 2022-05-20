import itertools
from pathlib import Path
import json
from copy import deepcopy

ROOT_PATH = './configs/decompose/tune/lf_no_warp_v2/'
root_dir = Path(ROOT_PATH)
root_dir.mkdir(parents=True, exist_ok=True)

LOAD_CONF = True

if LOAD_CONF:
  with (root_dir / 'config.json').open('r') as f:
    tune_conf = json.load(f)
  params = tune_conf['params']
else:
  tune_conf = {}
  tune_conf['source_conf'] = './configs/decompose/train_kubric_template.gin'

  params = []

  # quick experiment or full run
  tune_conf['quick_exp'] = False

  # params.append({'text': "TrainConfig.blendw_loss_weight_schedule = {{\n\
  # 'type': 'exp_increase',\n\
  # 'initial_value': {},\n\
  # 'final_value': 0.1,\n\
  # 'num_steps': 75000,\n\
  # }}",
  #               'values': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.005]
  # })

  # params.append("TrainConfig.shadow_r_loss_weight = {{ \n \
  #   'type': 'linear', \n \
  #   'initial_value': {}, \n \
  #   'final_value': {}, \n \
  #   'num_steps': 100000, \n \
  # }}")
  # values.append([[0.05, 0.1], [0.05, 0.01]])

  params.append({'text': "TrainConfig.blendw_loss_skewness = {}\n",
                'values': [0.5, 1.0, 1.5, 2.0, 5.0, 10.0] # [2.0]
                })


  params.append({'text': "TrainConfig.blendw_area_loss_weight = {}\n",
                'values': [0.0001] # [0.0]
  })
  # values.append([0.0001])


  params.append({'text': "EvalConfig.num_train_eval = {}\n",
                'values': [50]
  })

  params.append({'text': "EvalConfig.num_test_eval = {}\n",
                'values': [0]
  })


  tune_conf['params'] = deepcopy(params)
  with (root_dir / 'config.json').open('w') as f:
    json.dump(tune_conf, f, indent=2)

if 'quick_exp' in tune_conf:
  params.append({'text': "max_steps = {}\n",
                'values': [20000 if tune_conf['quick_exp'] else 100000]
  })

  params.append({'text': "EvalConfig.niter_runtime_eval = {}\n",
                'values': [2000 if tune_conf['quick_exp'] else 25000]
  })

else:
    params.append({'text': "EvalConfig.niter_runtime_eval = {}\n",
                'values': [25000]
  })


ids = []
for i in range(len(params)):
  ids.append(list(range(len(params[i]['values']))))

choices = list(itertools.product(*ids))

configs = []
for choice in choices:
  config = ""
  config += f"include '{tune_conf['source_conf']}'\n\n"
  for i in range(len(params)):
    v = params[i]['values'][choice[i]]
    if isinstance(v, list):
      config += params[i]['text'].format(*v) + "\n"
    else:
      config += params[i]['text'].format(v) + "\n"
  configs.append(config)

for i in range(len(configs)):
  filepath = root_dir / f"{i:03d}.gin"
  with filepath.open("w") as f:
      f.write(configs[i])

  print(filepath)








