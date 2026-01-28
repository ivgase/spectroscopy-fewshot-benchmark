import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from preprocessing import PreprocessNIR

##########################################
#   Definición de MixedTask y MixedDataset
##########################################

class MixedTask(Dataset):
    def __init__(self, support, query, name, device='cuda', path='data/MixedDataset'):
        """
        support y query son tuplas (X, y, indices) que se cargan desde CSV.
        """
        self.support_x, self.support_y, self.support_idx = support
        self.query_x, self.query_y, self.query_idx = query
        self.name = name
        self.path = path

        # Convertir pandas DataFrame o numpy array a tensores
        if isinstance(self.support_x, pd.DataFrame):
            self.support_x = torch.tensor(self.support_x.values, dtype=torch.float32)
            self.support_y = torch.tensor(self.support_y.values, dtype=torch.float32)
            self.query_x   = torch.tensor(self.query_x.values, dtype=torch.float32)
            self.query_y   = torch.tensor(self.query_y.values, dtype=torch.float32)
        elif isinstance(self.support_x, np.ndarray):
            self.support_x = torch.tensor(self.support_x, dtype=torch.float32)
            self.support_y = torch.tensor(self.support_y, dtype=torch.float32)
            self.query_x   = torch.tensor(self.query_x, dtype=torch.float32)
            self.query_y   = torch.tensor(self.query_y, dtype=torch.float32)

        # Aseguramos dimensiones adecuadas (agregamos una dimensión extra si es necesario)
        if self.support_x.ndim == 2:
            self.support_x = self.support_x.unsqueeze(1)
        if self.query_x.ndim == 2:
            self.query_x = self.query_x.unsqueeze(1)
        if self.support_y.ndim == 1:
            self.support_y = self.support_y.unsqueeze(1)
        if self.query_y.ndim == 1:
            self.query_y = self.query_y.unsqueeze(1)

        self.device = device
        self.nir_length = self.support_x.shape[2]

    def __len__(self):
        # Puede definirse según el tamaño del conjunto de soporte (o query) si se quiere iterar sobre ellos
        return self.support_x.shape[0]

    def __getitem__(self, idx):
        # Permite indexar individualmente si se requiere
        return self.support_x[idx], self.support_y[idx], self.support_idx[idx]
    
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

            query_df = pd.read_csv(os.path.join(self.path, f"{self.name}/fixed_val_query_{queries}shots.csv"), index_col=0)
            query_idx = query_df['query_idx'].values

        return {'support_features': self.support_x[support_idx].to(self.device),
                'support_targets': self.support_y[support_idx].to(self.device),
                'query_features': self.query_x[query_idx].to(self.device),
                'query_targets': self.query_y[query_idx].to(self.device)}

    # def sample_episode(self, shots, queries=1, batch_size=32):
    #     """
    #     Para problemas de regresión: genera un episodio con batch_size secuencias,
    #     donde cada secuencia tiene (shots + queries) ejemplos, tomando aleatoriamente
    #     ejemplos de las particiones de soporte y consulta.
    #     """
    #     episodes_features = []
    #     episodes_targets = []
        
    #     for _ in range(batch_size):
    #         # Samplear del conjunto de soporte
    #         num_support = self.support_x.shape[0]
    #         if num_support >= shots:
    #             support_indices = np.random.choice(num_support, shots, replace=False)
    #         else:
    #             support_indices = np.random.choice(num_support, shots, replace=True)
    #         np.random.shuffle(support_indices)
            
    #         # Samplear del conjunto de query
    #         num_query = self.query_x.shape[0]
    #         if num_query >= queries:
    #             query_indices = np.random.choice(num_query, queries, replace=False)
    #         else:
    #             query_indices = np.random.choice(num_query, queries, replace=True)
            
    #         support_features = self.support_x[support_indices].to(self.device)
    #         support_targets  = self.support_y[support_indices].to(self.device)
    #         query_features   = self.query_x[query_indices].to(self.device)
    #         query_targets    = self.query_y[query_indices].to(self.device)
            
    #         # Concatenar: primero soporte y luego query, imitando el orden original
    #         episode_features = torch.cat([support_features, query_features], dim=0)
    #         episode_targets  = torch.cat([support_targets, query_targets], dim=0)
            
    #         episodes_features.append(episode_features)
    #         episodes_targets.append(episode_targets)
        
    #     # Apilar los episodios en el batch
    #     batch_features = torch.stack(episodes_features, dim=0)
    #     batch_targets  = torch.stack(episodes_targets, dim=0)
        
    #     # Reorganizar para tener (batch_size*(shots+queries), 1, length)
    #     batch_features = batch_features.view(-1, *batch_features.shape[2:])
    #     batch_targets  = batch_targets.view(-1, *batch_targets.shape[2:])
        
    #     return {'features': batch_features, 'targets': batch_targets}

    def sample_episode(self, shots, queries=1):
        """
        Para problemas de regresión: genera un episodio con batch_size secuencias,
        donde cada secuencia tiene (shots + queries) ejemplos, tomando aleatoriamente
        ejemplos de las particiones de soporte y consulta.
        """
        episodes_features = []
        episodes_targets = []
        
        # Samplear del conjunto de soporte
        num_support = self.support_x.shape[0]
        if num_support >= shots:
            support_indices = np.random.choice(num_support, shots, replace=False)
        else:
            support_indices = np.random.choice(num_support, shots, replace=True)
        np.random.shuffle(support_indices)
        
        # Samplear del conjunto de query
        num_query = self.query_x.shape[0]
        if num_query >= queries:
            query_indices = np.random.choice(num_query, queries, replace=False)
        else:
            query_indices = np.random.choice(num_query, queries, replace=True)
        
        support_features = self.support_x[support_indices].to(self.device)
        support_targets  = self.support_y[support_indices].to(self.device)
        query_features   = self.query_x[query_indices].to(self.device)
        query_targets    = self.query_y[query_indices].to(self.device)
        
        episodes_features = []
        for idx in range(query_features.shape[0]):
            episode_features = torch.cat([support_features, query_features[idx].unsqueeze(0)], dim=0)
            episode_targets  = torch.cat([support_targets, query_targets[idx].unsqueeze(0)], dim=0)
            episodes_features.append(episode_features)
            episodes_targets.append(episode_targets)
        
        # Apilar los episodios en el batch
        batch_features = torch.stack(episodes_features, dim=0)
        batch_targets  = torch.stack(episodes_targets, dim=0)
        
        # Reorganizar para tener (batch_size*(shots+queries), 1, length)
        batch_features = batch_features.view(-1, *batch_features.shape[2:])
        batch_targets  = batch_targets.view(-1, *batch_targets.shape[2:])
        
        return {'features': batch_features, 'targets': batch_targets}

    
    def sample_episode_fixed(self, shots, queries, batch_size=32):
        """
        Para problemas de regresión: genera un episodio con batch_size secuencias,
        donde cada secuencia tiene (shots + queries) ejemplos, tomando aleatoriamente
        ejemplos de las particiones de soporte y consulta.
        """
        episodes_features = []
        episodes_targets = []
        
        samples = self.sample_fixed(shots, queries)        
        support_features = samples["support_features"]
        support_targets  = samples["support_targets"]
        query_features   = samples["query_features"]
        query_targets    = samples["query_targets"]

        for idx in range(query_features.shape[0]):
            episode_features = torch.cat([support_features, query_features[idx].unsqueeze(0)], dim=0)
            episode_targets  = torch.cat([support_targets, query_targets[idx].unsqueeze(0)], dim=0)
            episodes_features.append(episode_features)
            episodes_targets.append(episode_targets)
        
        # Apilar los episodios en el batch
        batch_features = torch.stack(episodes_features, dim=0)
        batch_targets  = torch.stack(episodes_targets, dim=0)
        
        # Dividir en batches de queries de tamaño 'batch_size'
        features_batches = torch.split(batch_features, batch_size, dim=0)
        targets_batches  = torch.split(batch_targets, batch_size, dim=0)

        features_batches = [fb.view(-1, *fb.shape[2:]) for fb in features_batches]
        targets_batches  = [tb.view(-1, *tb.shape[2:]) for tb in targets_batches]

        # # Reorganizar para tener (batch_size*(shots+queries), 1, length)
        # batch_features = batch_features.view(-1, *batch_features.shape[2:])
        # batch_targets  = batch_targets.view(-1, *batch_targets.shape[2:])
        
        return {'features': features_batches, 'targets': targets_batches}, support_features.shape[0]


    # def sample_episode(self, shots, queries=1, batch_size=32):
    #     """
    #     Para problemas de regresión: genera un episodio como una única secuencia,
    #     en la que los primeros 'shots' ejemplos (soporte) provienen del conjunto de soporte,
    #     y los últimos 'queries' ejemplos (consulta) provienen del conjunto de query.
    #     """
    #     # Samplear del conjunto de soporte
    #     num_support = self.support_x.shape[0]
    #     if num_support >= shots:
    #         support_indices = np.random.choice(num_support, shots, replace=False)
    #     else:
    #         support_indices = np.random.choice(num_support, shots, replace=True)
    #     np.random.shuffle(support_indices)
        
    #     # Samplear del conjunto de query
    #     num_query = self.query_x.shape[0]
    #     if num_query >= queries:
    #         query_indices = np.random.choice(num_query, queries, replace=False)
    #     else:
    #         query_indices = np.random.choice(num_query, queries, replace=True)
        
    #     support_features = self.support_x[support_indices].to(self.device)
    #     support_targets  = self.support_y[support_indices].to(self.device)
    #     query_features   = self.query_x[query_indices].to(self.device)
    #     query_targets    = self.query_y[query_indices].to(self.device)
        
    #     # Concatenar: primero soporte y luego query, imitando el orden original
    #     episode_features = torch.cat([support_features, query_features], dim=0)
    #     episode_targets  = torch.cat([support_targets, query_targets], dim=0)
        
    #     return {'features': episode_features, 'targets': episode_targets}

    
    def apply_padding(self, target_length):
        # Convertir support_x y query_x a arrays numpy removiendo la dimensión de canal (si existe)
        if self.support_x.ndim == 3:
            support_x_np = self.support_x.squeeze(1).cpu().numpy()
        else:
            support_x_np = self.support_x.cpu().numpy()
        if self.query_x.ndim == 3:
            query_x_np = self.query_x.squeeze(1).cpu().numpy()
        else:
            query_x_np = self.query_x.cpu().numpy()
        
        # Aplicar padding o recorte a cada secuencia para que tenga target_length
        support_x_padded = np.array([
            np.pad(x, (0, max(0, target_length - len(x))), mode='constant', constant_values=0)[:target_length]
            for x in support_x_np
        ])
        query_x_padded = np.array([
            np.pad(x, (0, max(0, target_length - len(x))), mode='constant', constant_values=0)[:target_length]
            for x in query_x_np
        ])
        
        # Convertir a tensores y reintroducir la dimensión de canal si es necesario
        support_x_tensor = torch.tensor(support_x_padded, dtype=torch.float32)
        query_x_tensor   = torch.tensor(query_x_padded, dtype=torch.float32)
        if support_x_tensor.ndim == 2:
            support_x_tensor = support_x_tensor.unsqueeze(1)
        if query_x_tensor.ndim == 2:
            query_x_tensor = query_x_tensor.unsqueeze(1)
        
        # Actualizar atributos y la longitud de NIR
        self.support_x = support_x_tensor.to(self.device)
        self.query_x   = query_x_tensor.to(self.device)
        if self.support_x.ndim >= 3:
            self.nir_length = self.support_x.shape[2]
        else:
            self.nir_length = self.support_x.shape[1]

class MixedDataset(Dataset):
    def __init__(self, path='data/MixedDataset', split='train', preprocessor=None, device='cuda'):
        self.device = device
        self.split = split        
        self.dataset_path = path

        split_df = pd.read_csv(os.path.join(self.dataset_path, "splits.csv"), index_col=0)
        tasks_names_split = split_df.query(f"split == '{split}'")['task'].values.tolist()

        self.tasks, self.task_names = self.__create_tasks(preprocessor=preprocessor,
                                                          path_data=self.dataset_path,
                                                          tasks_names_split=tasks_names_split)

    def __len__(self):
        return len(self.tasks)
    
    def max_length(self):
        return max(task.nir_length for task in self.tasks.values()
                   if isinstance(task.nir_length, int))
    
    def __create_tasks(self, preprocessor=None, path_data=None, tasks_names_split=None):
        tasks = {}
        for dataset in os.listdir(path_data):
            if dataset.strip() in tasks_names_split:
                support_x = pd.read_csv(os.path.join(path_data, dataset, "X_supp.csv"), index_col=0)
                support_y = pd.read_csv(os.path.join(path_data, dataset, "y_supp.csv"), index_col=0)
                query_x   = pd.read_csv(os.path.join(path_data, dataset, "X_query.csv"), index_col=0)
                query_y   = pd.read_csv(os.path.join(path_data, dataset, "y_query.csv"), index_col=0)
                support_idx = support_x.index
                query_idx   = query_x.index

                if preprocessor is not None:
                    support_x, support_y = preprocessor.fit_transform(support_x, support_y)
                    query_x, query_y     = preprocessor.transform(query_x, query_y)

                tasks[dataset] = MixedTask(support=(support_x, support_y, support_idx), 
                                           query=(query_x, query_y, query_idx), 
                                           name=dataset, path=self.dataset_path, device=self.device)
        return tasks, list(tasks.keys())
    
    def __getitem__(self, idx):
        name = self.task_names[idx]
        return self.tasks[name]
    
    def sample(self, shots, queries=1):
        # Devuelve un episodio sampleado de una tarea aleatoria
        task = np.random.choice(list(self.tasks.values()))
        return task.sample_episode(shots, queries)

##########################################
#   Dataset para episodios (envoltorio)
##########################################

class RegressionEpisodeDataset(Dataset):
    """
    Dataset que envuelve un conjunto de tareas (MixedDataset) y que, en cada llamada,
    devuelve un episodio sampleado usando el método sample_episode.
    """
    def __init__(self, tasks, shots, queries=1, batch_size=5):
        # tasks es un diccionario de tareas; convertimos a lista para indexar
        self.tasks = list(tasks.values())
        self.shots = shots
        self.queries = queries
        self.batch_size = batch_size

    def __len__(self):
        # Número de episodios igual al número de tareas (o se puede definir otra estrategia)
        return len(self.tasks)
    
    def __getitem__(self, idx):
        task = self.tasks[idx]
        return task.sample_episode(self.shots, self.queries)

    # def __getitem__(self, idx):
    #     task = self.tasks[idx]
    #     return task.sample_episode(self.shots, self.queries)

##########################################
#   Función de inicialización de DataLoaders
##########################################

def init_regression_dataset(opt, preprocessor):
    """
    Inicializa el MixedDataset y un DataLoader que devuelve episodios para problemas de regresión.
    
    opt debe tener los siguientes atributos:
      - opt.dataset: ruta a la carpeta de datos.
      - opt.split: 'train', 'val' o 'test'.
      - opt.shots: número de ejemplos de soporte.
      - opt.queries: número de ejemplos de query.
      - opt.batch_size: número de episodios por batch.
      - opt.device: 'cuda' o 'cpu'.
    """
    tr_dataset = MixedDataset(path=opt.dataset, split="train", preprocessor=preprocessor)
    val_dataset = MixedDataset(path=opt.dataset, split="val", preprocessor=preprocessor)
    test_dataset = MixedDataset(path=opt.dataset, split="test", preprocessor=preprocessor)
    max_length = max(tr_dataset.max_length(), val_dataset.max_length(), test_dataset.max_length())
    for task in tr_dataset.tasks.values():
        task.apply_padding(max_length)
    for task in val_dataset.tasks.values():
        task.apply_padding(max_length)
    for task in test_dataset.tasks.values():
        task.apply_padding(max_length)
    tr_episode_dataset = RegressionEpisodeDataset(tr_dataset.tasks, shots=opt.shots, queries=opt.queries, batch_size=opt.batch_size)
    val_episode_dataset = RegressionEpisodeDataset(val_dataset.tasks, shots=opt.shots, queries=opt.queries, batch_size=opt.batch_size)
    test_episode_dataset = RegressionEpisodeDataset(test_dataset.tasks, shots=opt.shots, queries=opt.queries, batch_size=opt.batch_size)
    tr_loader = DataLoader(tr_episode_dataset, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_episode_dataset, batch_size=opt.batch_size, shuffle=True)
    test_loader = DataLoader(test_episode_dataset, batch_size=opt.batch_size, shuffle=True)
    return tr_loader, val_loader, test_loader, tr_dataset, val_dataset, test_dataset

##########################################
#   Ejemplo de uso
##########################################

class Options:
    def __init__(self):
        self.path = 'data/MixedDataset'
        self.split = 'train'
        self.shots = 5      # Ejemplos de soporte por episodio
        self.queries = 1    # Ejemplos de query por episodio (puede ser mayor si lo deseas)
        self.batch_size = 2  # Número de episodios por batch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    opt = Options()
    preprocessor = PreprocessNIR(savgol=True, scale=False, scale_y=True, window_length=15, polyorder=2, deriv=1)
    dataset, loader = init_regression_dataset(opt, preprocessor)
    print("Tareas disponibles:", dataset.task_names)
    # Ejemplo: iterar sobre un batch de episodios
    for episode_batch in loader:
        # Cada elemento del batch es un diccionario con support_features, support_targets,
        # query_features y query_targets, de tamaño (batch_size, ...)
        print("Batch de episodios:")
        print(episode_batch['features'].shape, episode_batch['targets'].shape)
        break
