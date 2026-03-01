import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from abc import ABC, abstractmethod

#%% Loader Classes
class DataSource():
    def __init__(self, path:str):
        self.path = path
    


class Binance(DataSource):
    def __init__(self, path):
        super().__init__(path)
        self.data = pd.read_csv(self.path)

class Reddit(DataSource):
    def __init__(self, path):
        super().__init__(path)
        self.data = pd.read_csv(self.path)
    def get_num_users(self):
       return pd.concat([self.data['from'], self.data['to']]).nunique()
    def get_num_posts(self):
        return (self.data['post.id'] == self.data['comment.id']).sum()
    def get_num_comments(self):
        return (self.data['post.id'] != self.data['comment.id']).sum()
    def get_self_posts(self):
        return (self.data['from'] == self.data['to']).sum()

class Network(DataSource):
    def __init__(self,path):
        import pickle        
        super().__init__(path)
        with open(self.path, "rb") as f:
            self.data = pickle.load(f)

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




def get_deg_cent():
    # degree centrality
    deg_cent = nx.degree_centrality(G)
    return(deg_cent)

def get_bet_cent():
    # betweenness centrality (sample if graph is large)
    bet_cent = nx.betweenness_centrality(G, k=100, seed=42)
    return(bet_cent)



def plot(x1, x1_label, y1, y1_label, plot_label = 'Some Plot Label', title="some title", color1 = 'blue' plot2 = False, x2, y2, plot_label2 = 'Some Plot Label', color2='orange', log=False):
    plt.figure(figsize=(14,5))
    plt.plot(x,y,plot_label, color=color1)
    if plot2:
        plt.plot(x,y,label_scnd_plot, color=color2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if log:
        plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()




# list of moderators
mod_list = ['42points', 'Jools1802', 'GoodShibe', 'jimjunkdude', 'FloodgatesBot', 'RepostSleuthBot', 'AutoModerator']
reddit["is_moderator"] = reddit["from"].isin(mod_list)
reddit["to_moderator"] = reddit["to"].isin(mod_list)
reddit[reddit['is_moderator'] == 1]['from'].value_counts()
reddit[reddit['to_moderator'] == 1]['to'].value_counts()
raw_posts_only = reddit[(reddit['post.id'] == reddit['comment.id']) & (reddit['post.id']== reddit['parent.id'])]
raw_posts_only[raw_posts_only['is_moderator'] == 1]
# % of edges(posts/comments) (which involve mods)
mod_edge_share = (
    (reddit["is_moderator"] | reddit["to_moderator"]).mean()
)
all_nodes = pd.unique(pd.concat([reddit["from"], reddit["to"]]))
mod_nodes = reddit.loc[reddit["is_moderator"], "from"].unique()
# % of nodes (users) who are mods
mod_node_share = len(mod_nodes) / len(all_nodes)

float(mod_edge_share*100), mod_node_share * 100
reddit[reddit["is_moderator"]].groupby("from")["utc"].agg(["min","max"])


reddit['date'] = pd.to_datetime(reddit['time'], errors='coerce')
reddit = reddit.dropna(subset=['date'])
reddit['day'] = reddit['date'].dt.normalize()

daily_mod_counts = reddit[reddit['is_moderator']].groupby('day')['is_moderator'].count()
daily_nonmod_counts = reddit[~reddit['is_moderator']].groupby('day')['is_moderator'].count()

def create_activity():
# merge
    daily_activity = pd.concat([daily_mod_counts, daily_nonmod_counts], axis=1)
    daily_activity.columns = ['mod_activity', 'non_mod_activity']
    daily_activity = daily_activity.fillna(0)
    print(daily_activity.head())




#%% ###### EXECUTION ###########

reddit = Reddit(path='data/final_sorted.csv')
reddit_df = reddit.data
binance = Binance(path='data/Binance_DOGEUSDT_1h.csv')
binance_df = binance.data
G = Network(path="data/user_interaction_network.pkl")

#%%
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
deg_cent = nx.degree_centrality(G)

# Betweenness centrality (sample if graph is large)
bet_cent = nx.betweenness_centrality(G, k=100, seed=42)

# Show top users
top_deg = sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:10]
top_bet = sorted(bet_cent.items(), key=lambda x: x[1], reverse=True)[:10]

print("Top degree centrality:", top_deg)
print("Top betweenness centrality:", top_bet)


#%%%
# making plots
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

