import glob
import os
import requests
import json

def findFiles(path): return glob.glob(path)

collected_gid_eid_dict = {}
for sp in ['train','test','val']:

	path = './pbp_videos/%s/*'%sp
	local_path = '.'
	collected_videos = list(set(findFiles(path)))

	collected_urls = []
	for c_v in collected_videos:
    		collected_gid_eid_dict[c_v.split('/')[-1][:-4]] = True
    		#url = 'https://videos.nba.com/nba/pbp/media/'+a[0]+'/'+a[1]+'/'+a[2]+'/'+a[3]+'/'+a[4]+'/'+a[5]+'-'+a[6]+'-'+a[7]+'-'+a[8]+'-'+a[9]
    		#collected_urls.append(url)

urls = []
url2gameinfo = {}
game_eventid2video_id = json.load(open('%s/gameid_eventid2vid.json'%local_path, 'r'))

video_id2game_eventid = {v:k for k,v in game_eventid2video_id.items()}

split_dict = json.load(
    open('./split2video_id_after_videos_combination.json', 'r'))
videoid2split = {y: x for x in split_dict.keys() for y in split_dict[x]}
with open('./non_overlap_video_gid_eid.json', 'r') as f:
    non_overlap_video_gid_eid = json.load(f)
with open('./date_gameid_eventid_uuid.txt' , 'r') as fb:

    for fid, fbs in enumerate(fb):
        gameinfo = fbs.strip().split('\t')
        #print(gameinfo)
        if non_overlap_video_gid_eid.get(gameinfo[1]+'-'+gameinfo[2]) == None:
            continue
        if collected_gid_eid_dict.get(gameinfo[1]+'-'+ gameinfo[2])!= None:
            continue
        url = 'https://videos.nba.com/nba/pbp/media/'+gameinfo[0] +'/' + gameinfo[1] +'/' + gameinfo[2] + '/' +gameinfo[3]+'_1280x720.mp4'
        urls.append(url)
        url2gameinfo[url] = gameinfo

#urls = list(set(urls) - set(collected_urls))

# gameid = '0021800215'
# url_for_one_game = []
# for url in urls:
#     if gameid not in url:
#         continue
#     url_for_one_game.append(url)
#
# urls = url_for_one_game
print('%s videos remains'%len(urls))

#assert 1==0

for url in urls:
    print(url)
    r = requests.get(url, allow_redirects=True)
    gameinfo = url2gameinfo[url]
    vid = game_eventid2video_id[gameinfo[1]+'-'+gameinfo[2]]
    sp = videoid2split[vid]	
    print('start saving...')
    open('./pbp_videos/%s/%s.mp4'%(sp,gameinfo[1] +'-' + gameinfo[2]), 'wb').write(r.content)
