#%%
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from abc import ABC, abstractmethod
import pickle

from torch_geometric.data import Data
import torch
#%% Loader Classes
class DataSource():
    def __init__(self, path:str):
        self.path = path
    


class Binance(DataSource):
    def __init__(self, path = 'dataset/Binance_DOGEUSDT_1h.csv'):
        super().__init__(path)
        self.data = pd.read_csv(self.path)

class Reddit(DataSource):
    def __init__(self, path = 'dataset/final_sorted.csv'):
        super().__init__(path)
        self.data = pd.read_csv(self.path)
        self.set_moderator_status()
    def get_num_users(self):
       return pd.concat([self.data['from'], self.data['to']]).nunique()
    def get_num_posts(self):
        return (self.data['post.id'] == self.data['comment.id']).sum()
    def get_num_comments(self):
        return (self.data['post.id'] != self.data['comment.id']).sum()
    def get_self_posts(self):
        return (self.data['from'] == self.data['to']).sum()
    def set_moderator_status(self, mod_list = ['42points', 'Jools1802', 'GoodShibe', 'jimjunkdude', 'FloodgatesBot', 'RepostSleuthBot', 'AutoModerator']):
        """Sets moderator flags given a mod list"""
        self.data["from_moderator"] = self.data["from"].isin(mod_list)
        self.data["to_moderator"] = self.data["to"].isin(mod_list)


class Network(DataSource):
    def __init__(self,path):
        import pickle 
        import os       
        super().__init__(path)    
        # Use context manager to properly close the file
        with open(self.path, "rb") as f:
            self.data = pickle.load(f)
    def get_deg_cent(self):
        return nx.degree_centrality(self.data)
    def get_bet_cent(self):
        # betweenness centrality (sample if graph is large)
        return(nx.betweenness_centrality(self.data, k=100, seed=42))


#%%% Plotters 

class RedditPlotter:
    """Plot Reddit-specific metrics"""
    
    def __init__(self, figsize=(14, 5)):
        self.figsize = figsize
    
    def mod_activity(self, timestamps, mod_activity, 
                     title='Reddit Activity Over Time: Moderators vs Non-Moderators') -> plt.Figure:
        """Moderator activity only"""
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(timestamps, mod_activity, label='Moderator Activity', color='blue')
        
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Posts/Comments')
        ax.legend()
        
        fig.tight_layout()
        return fig
    
    def mod_vs_nonmod(self, timestamps, mod_activity, nonmod_activity,
                      title='Reddit Activity Over Time: Moderators vs Non-Moderators') -> plt.Figure:
        """Moderator vs Non-Moderator activity"""
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(timestamps, mod_activity, label='Moderator Activity', color='blue')
        ax.plot(timestamps, nonmod_activity, label='Non-Moderator Activity', color='orange', alpha=0.7)
        
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Posts/Comments')
        ax.legend()
        
        fig.tight_layout()
        return fig
    
    def mod_vs_nonmod_log(self, timestamps, mod_activity, nonmod_activity,
                          title='Reddit Activity Over Time: Moderators vs Non-Moderators') -> plt.Figure:
        """Moderator vs Non-Moderator activity (log scale)"""
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(timestamps, mod_activity, label='Moderator Activity', color='blue')
        ax.plot(timestamps, nonmod_activity, label='Non-Moderator Activity', color='orange', alpha=0.7)
        ax.set_yscale('log')
        
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Posts/Comments')
        ax.legend()
        
        fig.tight_layout()
        return fig


class BinancePlotter:
    """Plot Binance price data"""
    
    def __init__(self, figsize=(14, 6)):
        self.figsize = figsize
    
    def closing_price(self, timestamps, close_price,
                      title='Dogecoin Closing Price (Binance)') -> plt.Figure:
        """Closing price over time"""
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(timestamps, close_price, color='green')
        
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price (USD)')
        
        fig.tight_layout()
        return fig

def networkx_to_pyg(G, node_features=None):
    """Convert your NetworkX graph to PyG format"""
    # Create node mapping
    node_list = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    # Create edge index
    edge_list = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Node features (if none provided, use degree)
    if node_features is None:
        degrees = [G.degree(node) for node in node_list]
        x = torch.tensor(degrees, dtype=torch.float).view(-1, 1)
    else:
        x = node_features
    
    data = Data(x=x, edge_index=edge_index, num_nodes=len(node_list))
    return data, node_to_idx

class CombinedPlotter:
    """Plot Reddit and Binance data together"""
    
    def __init__(self, figsize=(16, 8)):
        self.figsize = figsize
    
    def price_and_mod_activity(self, price_timestamps, close_price, 
                               activity_timestamps, mod_activity,
                               title='Dogecoin Price & Moderator Activity Over Time') -> plt.Figure:
        """Price on top, moderator activity on bottom (stacked, shared x-axis)"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        
        # Top: Price
        ax1.plot(price_timestamps, close_price, color='green')
        ax1.set_ylabel('DOGE Price (USD)')
        ax1.set_title(title)
        
        # Bottom: Moderator activity
        ax2.plot(activity_timestamps, mod_activity, label='Moderator Activity', color='blue')
        ax2.set_ylabel('Moderator Activity')
        ax2.set_xlabel('Date')
        ax2.legend()
        
        fig.tight_layout()
        return fig

def load_pickle(file_path):
    # load pickle from filepath
    with open(file_path, 'rb') as f:
        output = pickle.load(f)
    return output
#%% 
if __name__ == '__main__':
    # list of moderators
    mod_list = ['42points', 'Jools1802', 'GoodShibe', 'jimjunkdude', 'FloodgatesBot', 'RepostSleuthBot', 'AutoModerator']
    
    # Load data
    reddit = Reddit(path='data/final_sorted.csv')
    reddit_df = reddit.data
    
    binance = Binance(path='data/Binance_DOGEUSDT_1h.csv')
    binance_df = binance.data
    G = Network(path="data/user_interaction_network.pkl")
    
    # Number of unique users
    n_users = reddit.get_num_users()
    print("Unique users:", n_users)

    # Number of posts vs comments
    n_posts = reddit.get_num_posts()
    n_comments = reddit.get_num_comments()
    print("Posts:", n_posts)
    print("Comments:", n_comments)

    # Self-posts (new threads)
    self_posts = reddit.get_self_posts()
    print("Thread-starting posts:", self_posts)

    # Degree centrality
    deg_cent = nx.degree_centrality(G.data)

    # Betweenness centrality (sample if graph is large)
    bet_cent = nx.betweenness_centrality(G.data, k=100, seed=42)

    # Show top users
    top_deg = sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:10]
    top_bet = sorted(bet_cent.items(), key=lambda x: x[1], reverse=True)[:10]

    print("Top degree centrality:", top_deg)
    print("Top betweenness centrality:", top_bet)

    # Moderator analysis
    reddit_df["is_moderator"] = reddit_df["from"].isin(mod_list)
    reddit_df["to_moderator"] = reddit_df["to"].isin(mod_list)
    reddit_df[reddit_df['is_moderator'] == 1]['from'].value_counts()
    reddit_df[reddit_df['to_moderator'] == 1]['to'].value_counts()
    raw_posts_only = reddit_df[(reddit_df['post.id'] == reddit_df['comment.id']) & (reddit_df['post.id']== reddit_df['parent.id'])]
    raw_posts_only[raw_posts_only['is_moderator'] == 1]
    
    # % of edges(posts/comments) (which involve mods)
    mod_edge_share = (
        (reddit_df["is_moderator"] | reddit_df["to_moderator"]).mean()
    )
    all_nodes = pd.unique(pd.concat([reddit_df["from"], reddit_df["to"]]))
    mod_nodes = reddit_df.loc[reddit_df["is_moderator"], "from"].unique()
    # % of nodes (users) who are mods
    mod_node_share = len(mod_nodes) / len(all_nodes)

    print(float(mod_edge_share*100), mod_node_share * 100)
    reddit_df[reddit_df["is_moderator"]].groupby("from")["time"].agg(["min","max"])

    # Create daily activity dataframe
    reddit_df['date'] = pd.to_datetime(reddit_df['time'], errors='coerce')
    reddit_df = reddit_df.dropna(subset=['date'])
    reddit_df['day'] = reddit_df['date'].dt.normalize()

    daily_mod_counts = reddit_df[reddit_df['is_moderator']].groupby('day')['is_moderator'].count()
    daily_nonmod_counts = reddit_df[~reddit_df['is_moderator']].groupby('day')['is_moderator'].count()

    def create_activity():
        # merge
        daily_activity = pd.concat([daily_mod_counts, daily_nonmod_counts], axis=1)
        daily_activity.columns = ['mod_activity', 'non_mod_activity']
        daily_activity = daily_activity.fillna(0)
        print(daily_activity.head())
        return daily_activity
    
    daily_activity = create_activity()
    
    # Prepare price data
    price_df = binance_df[['Date', 'Close']].copy()
    price_df['Date'] = pd.to_datetime(price_df['Date'], errors='coerce')
    
    # Making plots
    reddit_plotter = RedditPlotter()
    fig1 = reddit_plotter.mod_activity(daily_activity.index, daily_activity['mod_activity'])
    plt.show()

    fig2 = reddit_plotter.mod_vs_nonmod(
        daily_activity.index, 
        daily_activity['mod_activity'], 
        daily_activity['non_mod_activity']
    )
    plt.show()

    fig3 = reddit_plotter.mod_vs_nonmod_log(
        daily_activity.index, 
        daily_activity['mod_activity'], 
        daily_activity['non_mod_activity']
    )
    plt.show()

    binance_plotter = BinancePlotter()
    fig4 = binance_plotter.closing_price(price_df['Date'], price_df['Close'])
    plt.show()

    combined_plotter = CombinedPlotter()
    fig5 = combined_plotter.price_and_mod_activity(
        price_df['Date'], 
        price_df['Close'],
        daily_activity.index, 
        daily_activity['mod_activity']
    )
    plt.show()

# %%
