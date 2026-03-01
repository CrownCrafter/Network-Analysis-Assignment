import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np



from abc import ABC, abstractmethod

#%% Loader Classes
class DataSource(ABC):
    def __init__(self, path:str):
        self.path = path
    
    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass


class RedditLoader(DataSource):
    def __init__(self, path):
        super().__init__(path)
        self.data = None
    def load(self):
        self.data = pd.read_csv(self.path)
        return self.data

class BinanceLoader(DataSource):
    def __init__(self, path):
        self.data = None
        super().__init__(path)
    def load(self):
        self.data = pd.read_csv(self.path)
        return (self.data)
    def get_num_users(self):
       return pd.concat([self.data['from'], self.data['to']]).nunique()
    def get_num_posts(self):
        return (self.data['post.id'] == self.data['comment.id']).sum()
    def get_num_comments(self):
        return (self.data['post.id'] != self.data['comment.id']).sum()
    def get_self_posts(self):
        return (self.data['from'] == self.data['to']).sum()

class NetworkLoader(RedditLoader):
    def __init__(self,path, data):
        super().__init__(path)
        self.data = None
    def load(self):
        import pickle
        with open(self.path, "wb") as f:
            pickle.dump(self.data, f)
        with open(self.path, "rb") as f:
            self.data = pickle.load(f)




def get_deg_cent():
    # degree centrality
    deg_cent = nx.degree_centrality(G)
    return(deg_cent)

def get_bet_cent():
    # betweenness centrality (sample if graph is large)
    bet_cent = nx.betweenness_centrality(G, k=100, seed=42)
    return(bet_cent)

class Plotter(ABC):
    def __init__():


def 


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



#%%
class Graph():
    def __init__(self):
        

class Degrees(Graph):
    def __init__(self, degrees:np.ndarray):
        self.degrees = [deg for node, deg in Graph.degree]
    def get_median(self):
        return np.median(self.degrees)
    def get_mean(self):
        return np.mean(self.degrees)
    def get_top_degrees(self):
        top_degrees = sorted(self.degrees, key=lambda x: x[1], reverse=True)[:10]
        print("Top 10 users by degree:")
        for user, deg in top_degrees:
            print(user, deg)

    def plot_degree_dist(self):
        # Plot histogram (linear scale)
        plt.figure(figsize=(8,5))
        plt.boxplot(self.degrees)
        plt.title("Degree Distribution (User Interaction Network)")
        plt.xlabel("Degree (number of unique users interacted with)")
        plt.ylabel("Number of users")
        plt.show()
    def plot_degree_log_dist(self):
        plt.figure(figsize=(8,5))
        plt.hist(self.degrees, bins=30, color='salmon', edgecolor='black', log=True)
        plt.title("Degree Distribution (Log Scale)")
        plt.xlabel("Degree")
        plt.ylabel("Number of users (log scale)")
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




###### EXECUTION ###########

reddit = Dataloader(path='/data/final_sorted.csv')
binance = Dataloader(path='Binance_DOGEUSDT_1h.csv')





plot(x=daily_activity.index,xlabel='Date', y=daily_activity['mod_activity'], ylabel='Number of Posts/Comments', plot_label='Moderator Activity', title='Reddit Activity Over Time: Moderators vs Non-Moderators')

#plot_mod_activity
plot(daily_activity.index,'Date',daily_activity['mod_activity'], 'Number of Posts/Comments', plot_label='Moderator Activity',title='Reddit Activity Over Time: Moderators vs Non-Moderators',color1='blue',
        plot2=True, daily_activity.index, daily_activity['non_mod_activity'], plot_label2='Non-Moderator Activity',color2 = 'orange')

#plot_mod_log_activity
plot(daily_activity.index,'Date',daily_activity['mod_activity'], 'Number of Posts/Comments', plot_label='Moderator Activity',title='Reddit Activity Over Time: Moderators vs Non-Moderators',color1='blue',
        plot2=True, daily_activity.index, daily_activity['non_mod_activity'], plot_label2='Non-Moderator Activity',color2 = 'orange', log=True)



binance['Date'] = pd.to_datetime(binance['Date'], errors='coerce')
binance = binance.dropna(subset=['Date'])  # drop bad timestamps

# 2️⃣ Sort by date (important)
binance = binance.sort_values('Date')

# 3️⃣ Plot closing price over time
plot(price_reddit['Date'], 'Date', binance['Close'], 'Close Price (USD)', '', 'Dogecoin Closing Price (Binance)', color1='grren', plot2=False)



all_dates = pd.date_range(start=min(price_reddit.index.min(), daily_mod_counts.index.min()),
                        end=max(price_reddit.index.max(), daily_mod_counts.index.max()))
daily_mod_counts = daily_mod_counts.reindex(all_dates, fill_value=0)

def stacked_plots():
    # 2 stacked subplots with shared x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,8), sharex=True)

    # Top: Dogecoin price
    ax1.plot(price_reddit.index, price_reddit['Close'], color='green')
    ax1.set_ylabel('DOGE Price (USD)')
    ax1.set_title('Dogecoin Price & Moderator Activity Over Time')

    ax2.plot(daily_activity.index, daily_activity['mod_activity'], label='Moderator Activity', color='blue')
    ax2.set_ylabel('Moderator Activity')
    ax2.set_xlabel('date')

    plt.tight_layout()
    plt.show()
stacked_plots

# Get top 5 days with highest moderator activity
top_mod_days = daily_activity['mod_activity'].sort_values(ascending=False).head(5)
print(top_mod_days)
