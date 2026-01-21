import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from preprocessing import PreprocessNIR
from copy import deepcopy
    
class Task(Dataset):
    def __init__(self, data, region, prop, idx=None, device='cuda', split_fewshot=True, preprocessor=None, path_data=None):
        self.nir, self.targets = data
        self.idx = idx
        self.path_data = path_data
        self.prop = prop
        if split_fewshot:
            if path_data is None or not os.path.exists(os.path.join(path_data, f"split_{region}_{prop}")):
                support_x, query_x, support_y, query_y = train_test_split(self.nir, self.targets, test_size=0.5, random_state=42)
                if not os.path.exists(os.path.join(path_data, f"split_{region}_{prop}")):
                    os.makedirs(os.path.join(path_data, f"split_{region}_{prop}"), exist_ok=True)
                    support_x.to_csv(os.path.join(path_data, f"split_{region}_{prop}", "support_x.csv"))
                    query_x.to_csv(os.path.join(path_data, f"split_{region}_{prop}", "query_x.csv"))
                    support_y.to_csv(os.path.join(path_data, f"split_{region}_{prop}", "support_y.csv"))
                    query_y.to_csv(os.path.join(path_data, f"split_{region}_{prop}", "query_y.csv"))
            else:
                support_x = pd.read_csv(os.path.join(path_data, f"split_{region}_{prop}", "support_x.csv"), index_col=0)
                query_x = pd.read_csv(os.path.join(path_data, f"split_{region}_{prop}", "query_x.csv"), index_col=0)
                support_y = pd.read_csv(os.path.join(path_data, f"split_{region}_{prop}", "support_y.csv"), index_col=0)
                query_y = pd.read_csv(os.path.join(path_data, f"split_{region}_{prop}", "query_y.csv"), index_col=0)

            support_y = support_y[prop]
            query_y = query_y[prop]
            if isinstance(support_y, pd.DataFrame):
                nan_support = support_y.isna().any(axis=1)
                nan_query = query_y.isna().any(axis=1)
            elif isinstance(support_y, pd.Series):
                nan_support = support_y.isna()
                nan_query = query_y.isna()
            support_x = support_x[~nan_support]#.reset_index(drop=True)
            support_y = support_y[~nan_support]#.reset_index(drop=True)
            query_x = query_x[~nan_query]#.reset_index(drop=True)
            query_y = query_y[~nan_query]#.reset_index(drop=True)
            supp_idx = support_x.index
            query_idx = query_x.index
            support_x = support_x.reset_index(drop=True)
            support_y = support_y.reset_index(drop=True)
            query_x = query_x.reset_index(drop=True)
            query_y = query_y.reset_index(drop=True)

            assert(self.nir.shape[0] == support_x.shape[0] + query_x.shape[0])
            assert(self.targets.shape[0] == support_y.shape[0] + query_y.shape[0])
            assert(support_x.shape[0] == support_y.shape[0])
            assert(query_x.shape[0] == query_y.shape[0])
                                                      
            self.support_x = torch.tensor(support_x.values, dtype=torch.float32)
            self.support_y = torch.tensor(support_y.values, dtype=torch.float32)
            self.query_x = torch.tensor(query_x.values, dtype=torch.float32)
            self.query_y = torch.tensor(query_y.values, dtype=torch.float32)
            self.support_idx = supp_idx
            self.query_idx = query_idx
        else:
            if type(self.nir) == pd.DataFrame:
                self.support_x = torch.tensor(self.nir.values, dtype=torch.float32)
                self.support_y = torch.tensor(self.targets.values, dtype=torch.float32)
                self.query_x = torch.tensor(self.nir.values, dtype=torch.float32)
                self.query_y = torch.tensor(self.targets.values, dtype=torch.float32)
            else:
                self.support_x = torch.clone(self.nir)
                self.support_y = torch.clone(self.targets)
                self.query_x = torch.clone(self.nir)
                self.query_y = torch.clone(self.targets)

            if self.idx is not None:
                assert len(self.idx) == 2, "Index must be a tuple with two elements (support and query indexes)"
                self.support_idx, self.query_idx = self.idx

        if preprocessor is not None:
            preprocessor_copy = deepcopy(preprocessor)
            self.support_x, self.support_y = preprocessor_copy.fit_transform(self.support_x, self.support_y)
            self.query_x, self.query_y = preprocessor_copy.transform(self.query_x, self.query_y)

        if self.support_x.ndim == 2:
            self.support_x = self.support_x.unsqueeze(1)
        if self.query_x.ndim == 2:
            self.query_x = self.query_x.unsqueeze(1)
        if self.support_y.ndim == 1:
            self.support_y = self.support_y.unsqueeze(1)
        if self.query_y.ndim == 1:
            self.query_y = self.query_y.unsqueeze(1)

        self.region = region
        self.device = device

    def __len__(self):
        return self.nir.shape[0]
    
    def __getitem__(self, idx):
        return self.query_x[idx], self.query_y[idx], self.query_idx[idx]
    
    def sample(self, shots, queries):
        try:
            support_idx = np.random.choice(self.support_x.shape[0], shots, replace=False)
            query_idx = np.random.choice(self.query_x.shape[0], queries, replace=False)
        except ValueError:
            return None

        return {'support_features': self.support_x[support_idx].to(self.device),
                'support_targets': self.support_y[support_idx].to(self.device),
                'query_features': self.query_x[query_idx].to(self.device),
                'query_targets': self.query_y[query_idx].to(self.device)}

    def sample_fixed(self, shots, queries):
        if not os.path.exists(os.path.join(self.path_data, f"split_{self.region}_{self.prop}", f"fixed_val_{shots}shots.csv")):
            support_idx = np.random.choice(self.support_x.shape[0], shots, replace=False)
            query_idx = np.random.choice(self.query_x.shape[0], queries, replace=False)
            val = pd.DataFrame({'support_idx': support_idx, 'query_idx': query_idx})
            val.to_csv(os.path.join(self.path_data, f"split_{self.region}_{self.prop}", f"fixed_val_{shots}shots.csv"))
        else:
            val = pd.read_csv(os.path.join(self.path_data, f"split_{self.region}_{self.prop}", f"fixed_val_{shots}shots.csv"), index_col=0)
            support_idx = val.support_idx.values
            query_idx = val.query_idx.values

        return {'support_features': self.support_x[support_idx].to(self.device),
                'support_targets': self.support_y[support_idx].to(self.device),
                'query_features': self.query_x[query_idx].to(self.device),
                'query_targets': self.query_y[query_idx].to(self.device)}
    
    def query_dataloader(self):
        dataset = Task(data=(self.query_x, self.query_y), region=self.region, prop=self.prop, idx=(None, self.query_idx), split_fewshot=False)
        return DataLoader(dataset, batch_size=32, shuffle=False)
        
    
class SoilDataset(Dataset):
    def __init__(self, prop, split='train', data_type="nir", granularity_level="country", supp_sz=5, query_sz=5, preprocessor=None, device='cuda'):
        all_props = ['N', 'OC', 'CEC', 'pH_h2o', 'pH_cacl2', 'P', 'K', 'CaCO3', 'Clay', 'Silt', 'Sand']
        all_props_lower = [prop.lower() for prop in all_props]
        if isinstance(prop, list):
            for p in prop:
                assert p.lower() in all_props_lower, f"Prop {p} not found"
        elif isinstance(prop, str):
            assert prop.lower() in all_props_lower, f"Prop {prop} not found"
        assert split in ["train", "val", "test"], f"Split {split} not found"
        # if prop != 'K':
        #     raise NotImplementedError("Only Potassium (K) is implemented for now")

        self.split = split
        self.supp_sz = supp_sz
        self.query_sz = query_sz
        self.device = device
        self.prop = prop
        self.granularity_level = granularity_level

        # sizes = [50,100]
        # for size in sizes:
        #     if size >= supp_sz * 2:
        #         size_dataset = size
        #         break

        if data_type == "nir":
            dirname = "Soil_all"
            x_colnames = "scan_visnir"
        elif data_type == "mir":
            # dirname = "MIR_all"
            dirname = "MIR_by_climzone"
            x_colnames = "scan_mir"


        data = pd.read_csv(f"data/{dirname}/{split}_tasks.csv", index_col=0)
        prop_dropna = prop if isinstance(prop, list) else [prop]
        data = data.dropna(subset=prop_dropna, axis=0)#.reset_index(drop=True)
        x_columns = [col for col in data.columns if x_colnames in col]
        self.nir = data[x_columns]
        self.targets = data[prop]
        location_cols = ['Country']
        if "Continent" in data.columns:
            location_cols = location_cols + ['Continent']
        if "State" in data.columns:
            location_cols = location_cols + ['State']
        self.targets_w_country = data[prop_dropna + location_cols]

        self.tasks = self.__create_tasks(preprocessor=preprocessor, path_data=f"data/{dirname}/")
        self.task_names = list(self.tasks.keys())

    def __create_tasks(self, preprocessor=None, path_data=None):
        tasks = {}
        for country in self.targets_w_country[self.granularity_level].unique():
            country_idx = self.targets[self.targets_w_country[self.granularity_level] == country].index
            if len(country_idx) >= self.supp_sz + self.query_sz:
                tasks[country] = Task((self.nir.loc[country_idx], self.targets.loc[country_idx]), prop=self.prop,
                                    region=country, device=self.device, preprocessor=preprocessor, path_data=path_data)
        return tasks
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        name = self.task_names[idx]
        return self.tasks[name]
    
    def sample_task(self):
        task = np.random.choice(self.tasks)
        return task
    
    def sample(self):
        task = self.sample_task()
        return task.sample(self.supp_sz, self.query_sz)
    
    def query_dataloader(self):
        dataset = SoilDataset(prop='K', split=self.split, supp_sz=self.supp_sz, query_sz=self.query_sz)
        return DataLoader(dataset, batch_size=32, shuffle=False)

    def batch_dataloader(self, mode='support'):
        return DataLoader(BatchDataset(self.tasks, self.supp_sz, mode), batch_size=32, shuffle=True)


class BatchDataset(Dataset):
    def __init__(self, tasks, shots, mode='support'):
        self.tasks = tasks
        self.shots = shots

        x = []
        y = []
        for region in self.tasks.keys():
            if mode == 'support':
                x.append(self.tasks[region].support_x)
                y.append(self.tasks[region].support_y)
            elif mode == 'query':
                x.append(self.tasks[region].query_x)
                y.append(self.tasks[region].query_y)
            else:
                raise ValueError("Mode not found")

        shapes = [t.shape[-1] for t in x]
        if len(set(shapes)) > 1:
            max_shape = max(shapes)
            for i in range(len(x)):
                if x[i].shape[-1] < max_shape:
                    x[i] = torch.cat([x[i], torch.zeros(x[i].shape[0], 1, max_shape - x[i].shape[-1])], dim=-1)
        
        self.x = torch.cat(x, dim=0)
        self.y = torch.cat(y, dim=0)

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

class MangoQuery(Dataset):
    def __init__(self, data, preprocessor=None):
        self.nir, self.targets = data
        if preprocessor is not None:
            self.nir, self.targets = preprocessor.fit_transform(self.nir, self.targets)

        if type(self.nir) == pd.DataFrame:
            self.nir = torch.tensor(self.nir.values, dtype=torch.float32)
            self.targets = torch.tensor(self.targets.values, dtype=torch.float32)
        elif type(self.nir) == np.ndarray:
            self.nir = torch.tensor(self.nir, dtype=torch.float32)
            self.targets = torch.tensor(self.targets, dtype=torch.float32)

        if self.nir.ndim == 2:
            self.nir = self.nir.unsqueeze(1)
        if self.targets.ndim == 1:
            self.targets = self.targets.unsqueeze(1)

    def __len__(self):
        return self.nir.shape[0]
    
    def __getitem__(self, idx):
        return self.nir[idx], self.targets[idx]

    
class MangoDataset(Dataset):
    def __init__(self, supp_sz=5, query_sz=5, preprocessor=None, device='cuda'):
        self.supp_sz = supp_sz
        self.query_sz = query_sz
        self.device = device
        
        self.supp_x = pd.read_csv("data/Mango_new/support_x.csv", index_col=0)
        self.supp_y = pd.read_csv("data/Mango_new/support_y.csv", index_col=0)
        self.query_x = pd.read_csv("data/Mango_new/query_x.csv", index_col=0)
        self.query_y = pd.read_csv("data/Mango_new/query_y.csv", index_col=0)

        if preprocessor is not None:
            self.supp_x, self.supp_y = preprocessor.fit_transform(self.supp_x, self.supp_y)
            self.query_x, self.query_y = preprocessor.transform(self.query_x, self.query_y)

        if type(self.supp_x) == pd.DataFrame:
            self.supp_x = torch.tensor(self.supp_x.values, dtype=torch.float32)
            self.supp_y = torch.tensor(self.supp_y.values, dtype=torch.float32)
            self.query_x = torch.tensor(self.query_x.values, dtype=torch.float32)
            self.query_y = torch.tensor(self.query_y.values, dtype=torch.float32)
        elif type(self.supp_x) == np.ndarray:
            self.supp_x = torch.tensor(self.supp_x, dtype=torch.float32)
            self.supp_y = torch.tensor(self.supp_y, dtype=torch.float32)
            self.query_x = torch.tensor(self.query_x, dtype=torch.float32)
            self.query_y = torch.tensor(self.query_y, dtype=torch.float32)

        if self.supp_x.ndim == 2:
            self.supp_x = self.supp_x.unsqueeze(1)
        if self.query_x.ndim == 2:
            self.query_x = self.query_x.unsqueeze(1)
        if self.supp_y.ndim == 1:
            self.supp_y = self.supp_y.unsqueeze(1)
        if self.query_y.ndim == 1:
            self.query_y = self.query_y.unsqueeze(1)
    
    def __len__(self):
        return self.supp_x.shape[0] + self.query_x.shape[0]
    
    def __getitem__(self, idx):
        if idx < self.supp_x.shape[0]:
            return self.supp_x[idx], self.supp_y[idx]
        else:
            return self.query_x[idx - self.supp_x.shape[0]], self.query_y[idx - self.supp_x.shape[0]]
    
    def sample(self):
        support_idx = np.random.choice(self.supp_x.shape[0], self.supp_sz, replace=False)
        query_idx = np.random.choice(self.query_x.shape[0], self.query_sz, replace=False)

        return {'support_features': self.supp_x[support_idx].to(self.device),
                'support_targets': self.supp_y[support_idx].to(self.device),
                'query_features': self.query_x[query_idx].to(self.device),
                'query_targets': self.query_y[query_idx].to(self.device)}

    def sample_fixed(self):
        if not os.path.exists(f"data/Mango_new/fixed_val_{self.supp_sz}.csv"):
            support_idx = np.random.choice(self.supp_x.shape[0], self.supp_sz, replace=False)
            query_idx = np.random.choice(self.query_x.shape[0], self.query_sz, replace=False)
            val = pd.DataFrame({'support_idx': support_idx, 'query_idx': query_idx})
            val.to_csv(f"data/Mango_new/fixed_val_{self.supp_sz}.csv")
        else:
            val = pd.read_csv(f"data/Mango_new/fixed_val_{self.supp_sz}.csv", index_col=0)
            support_idx = val.support_idx.values
            query_idx = val.query_idx.values

        return {'support_features': self.supp_x[support_idx].to(self.device),
                'support_targets': self.supp_y[support_idx].to(self.device),
                'query_features': self.query_x[query_idx].to(self.device),
                'query_targets': self.query_y[query_idx].to(self.device)}
    
    def query_dataloader(self, batch_size=32):
        dataset = MangoQuery(data=(self.query_x, self.query_y))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

class MixedTask(Dataset):
    def __init__(self, support, query, name, device='cuda', path='data/MixedDataset', noise_aug=None):
        self.support_x, self.support_y, self.support_idx = support
        self.query_x, self.query_y, self.query_idx = query
        self.name = name
        self.path = path
        self.noise_aug = noise_aug

        if type(self.support_x) == pd.DataFrame:
            self.support_x = torch.tensor(self.support_x.values, dtype=torch.float32)
            self.support_y = torch.tensor(self.support_y.values, dtype=torch.float32)
            self.query_x = torch.tensor(self.query_x.values, dtype=torch.float32)
            self.query_y = torch.tensor(self.query_y.values, dtype=torch.float32)
        elif type(self.support_x) == np.ndarray:
            self.support_x = torch.tensor(self.support_x, dtype=torch.float32)
            self.support_y = torch.tensor(self.support_y, dtype=torch.float32)
            self.query_x = torch.tensor(self.query_x, dtype=torch.float32)
            self.query_y = torch.tensor(self.query_y, dtype=torch.float32)

        if self.support_x.ndim == 2:
            self.support_x = self.support_x.unsqueeze(1)
        if self.query_x.ndim == 2:
            self.query_x = self.query_x.unsqueeze(1)
        if self.support_y.ndim == 1:
            self.support_y = self.support_y.unsqueeze(1)
        if self.query_y.ndim == 1:
            self.query_y = self.query_y.unsqueeze(1)

        self.device = device

    def __len__(self):
        return self.query_x.shape[0]
    
    def __getitem__(self, idx):
        if self.noise_aug is not None:
            query_x = self.noise_aug(self.query_x[idx])
        else:
            query_x = self.query_x[idx]
        # query_x = self.query_x[idx]
        return query_x, self.query_y[idx], self.query_idx[idx]
    
    def sample(self, shots, queries):
        if self.support_x.shape[0] >= shots:
            support_idx = np.random.choice(self.support_x.shape[0], shots, replace=False)
        else:
            support_idx = np.random.choice(self.support_x.shape[0], self.support_x.shape[0], replace=False)

        if self.query_x.shape[0] >= queries:
            query_idx = np.random.choice(self.query_x.shape[0], queries, replace=False)
        else:
            query_idx = np.random.choice(self.query_x.shape[0], self.query_x.shape[0], replace=False)

        if self.noise_aug is not None:
            support_x = self.noise_aug(self.support_x[support_idx])
        else:
            support_x = self.support_x[support_idx]

        return {'support_features': support_x.to(self.device),
                'support_targets': self.support_y[support_idx].to(self.device),
                'query_features': self.query_x[query_idx].to(self.device),
                'query_targets': self.query_y[query_idx].to(self.device)}

    def sample_fixed(self, shots, queries):
        if not os.path.exists(os.path.join(self.path, f"{self.name}/fixed_val_support_{shots}shots.csv")) or not os.path.exists(os.path.join(self.path, f"{self.name}/fixed_val_support_{shots}shots.csv")):
            if self.support_x.shape[0] >= shots:
                support_idx = np.random.choice(self.support_x.shape[0], shots, replace=False)
            else:
                support_idx = np.random.choice(self.support_x.shape[0], self.support_x.shape[0], replace=False)

            if self.query_x.shape[0] >= queries:
                query_idx = np.random.choice(self.query_x.shape[0], queries, replace=False)
            else:
                query_idx = np.random.choice(self.query_x.shape[0], self.query_x.shape[0], replace=False)

            # val = pd.DataFrame({'support_idx': support_idx, 'query_idx': query_idx})
            # val.to_csv(os.path.join(self.path, f"{self.name}/fixed_val_{shots}shots.csv"))
            pd.DataFrame({'support_idx': support_idx}).to_csv(os.path.join(self.path, f"{self.name}/fixed_val_support_{shots}shots.csv"))
            pd.DataFrame({'query_idx': query_idx}).to_csv(os.path.join(self.path, f"{self.name}/fixed_val_query_{shots}shots.csv"))
        else:
            support_df = pd.read_csv(os.path.join(self.path, f"{self.name}/fixed_val_support_{shots}shots.csv"), index_col=0)
            support_idx = support_df['support_idx'].values

            query_df = pd.read_csv(os.path.join(self.path, f"{self.name}/fixed_val_query_{shots}shots.csv"), index_col=0)
            query_idx = query_df['query_idx'].values

        if self.noise_aug is not None:
            support_x = self.noise_aug(self.support_x[support_idx])
        else:
            support_x = self.support_x[support_idx]

        return {'support_features': support_x.to(self.device),
                'support_targets': self.support_y[support_idx].to(self.device),
                'query_features': self.query_x[query_idx].to(self.device),
                'query_targets': self.query_y[query_idx].to(self.device)}
    
    def query_dataloader(self):
        # dataset = MixedTask(support=(self.query_x, self.query_y, self.query_idx), query=(self.query_x, self.query_y, self.query_idx), name=self.name, noise_aug=self.noise_aug)
        dataset = MixedTask(support=(self.query_x, self.query_y, self.query_idx), query=(self.query_x, self.query_y, self.query_idx), name=self.name)
        return DataLoader(dataset, batch_size=32, shuffle=False)

    def get_shape(self, idx):
        return self.query_x[idx].shape
    

class MixedDataset(Dataset):
    def __init__(self, path='data/MixedDataset', split='train', supp_sz=5, query_sz=5, preprocessor=None, device='cuda', max_tasks=None, noise_aug=None):
        self.supp_sz = supp_sz
        self.query_sz = query_sz
        self.device = device
        self.split = split        
        self.dataset_path = path
        self.max_tasks = max_tasks

        self.noise_aug = noise_aug

        split_df = pd.read_csv(os.path.join(self.dataset_path, "splits.csv"), index_col=0)
        tasks_names_split = split_df.query(f"split == '{split}'")['task'].values.tolist()

        if self.max_tasks is not None:
            if os.path.exists(os.path.join(self.dataset_path, "choices.json")):
                import json
                with open(os.path.join(self.dataset_path, "choices.json"), 'r') as f:
                    choices = json.load(f)
                tasks_names_split = choices[str(self.max_tasks)]
            else:
                if len(tasks_names_split) > self.max_tasks:
                    tasks_names_split = np.random.choice(tasks_names_split, self.max_tasks, replace=False).tolist()
                else:
                    tasks_names_split = np.random.choice(tasks_names_split, len(tasks_names_split), replace=False).tolist()

        self.tasks, self.task_names = self.__create_tasks(preprocessor=preprocessor, path_data=self.dataset_path, tasks_names_split=tasks_names_split)

        # assert(len(self.tasks) == len(tasks_names_split))

    def __len__(self):
        return len(self.tasks)
    
    def __create_tasks(self, preprocessor=None, path_data=None, tasks_names_split=None):
        tasks = {}
        for dataset in os.listdir(path_data):
            if dataset.strip() in tasks_names_split:
                support_x = pd.read_csv(os.path.join(path_data, dataset, "X_supp.csv"), index_col=0)
                support_y = pd.read_csv(os.path.join(path_data, dataset, "y_supp.csv"), index_col=0)
                query_x = pd.read_csv(os.path.join(path_data, dataset, "X_query.csv"), index_col=0)
                query_y = pd.read_csv(os.path.join(path_data, dataset, "y_query.csv"), index_col=0)
                support_idx = support_x.index
                query_idx = query_x.index

                if preprocessor is not None:
                    support_x, support_y = preprocessor.fit_transform(support_x, support_y)
                    query_x, query_y = preprocessor.transform(query_x, query_y)

                tasks[dataset] = MixedTask(support=(support_x, support_y, support_idx), query=(query_x, query_y, query_idx), 
                                           name=dataset, path=self.dataset_path, noise_aug=self.noise_aug)
        return tasks, list(tasks.keys())
    
    def __getitem__(self, idx):
        name = self.task_names[idx]
        return self.tasks[name]
    
    def sample_task(self):
        task = np.random.choice(self.tasks)
        return task
    
    def sample(self):
        task = self.sample_task()
        return task.sample(self.supp_sz, self.query_sz)

    def batch_dataloader(self, mode='support'):
        return DataLoader(BatchDataset(self.tasks, self.supp_sz, mode), batch_size=32, shuffle=True)