from django.urls import path

from solsys_code.solsys_code_observatory.views import CreateObservatory, ObservatoryDetailView, ObservatoryList

app_name = 'solsys_code.solsys_code_observatory'

urlpatterns = [
    # path('create/<str:mpccode>/', CreateObservatory.as_view(), name='create'),
    path('create/', CreateObservatory.as_view(), name='create'),
    path('<int:pk>/', ObservatoryDetailView.as_view(), name='detail'),
    path('', ObservatoryList.as_view(), name='list'),
]
