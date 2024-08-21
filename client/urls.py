# music_player/urls.py

# from django.urls import path
# from client.views import index,play_emotion

# urlpatterns = [
#     path('', index, name='index'),
#     path('play/', play_emotion, name='play_emotion'),
# ]
from django.urls import path
from client import views

urlpatterns = [
    path('', views.index, name='index'),  # Root URL for index page
    path('start_capturing/', views.start_capturing, name='start_capturing'),
    path('detect_emotion/', views.detect_emotion, name='detect_emotion'),
    path('playlist_songs_by_emotion/<str:emotion>/', views.playlist_songs_by_emotion, name='playlist_songs_by_emotion'),
]
