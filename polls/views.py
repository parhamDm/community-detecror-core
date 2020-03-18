from django.http import JsonResponse
from django.shortcuts import render

# Create your views here.
from django.views.decorators.csrf import csrf_exempt

from polls.utils.detector import Detector



def edgebetweenness(request):
    dtct = Detector()
    id = request.GET["id"]
    # todo auth
    data = dtct.edge_betweenness(id)
    return JsonResponse(data, safe=False)

def fastgreedy(request):
    dtct = Detector()
    id = request.GET["id"]
    # todo auth
    data = dtct.fast_greedy(id)
    return JsonResponse(data, safe=False)

def walktrap(request):
    dtct = Detector()
    id = request.GET["id"]
    # todo auth
    data = dtct.walk_trap(id)
    return JsonResponse(data, safe=False)

@csrf_exempt
def register(request):
    dtct = Detector()
    body = request._get_post()
    id = request.GET["id"]
    data = dtct.registerGraph(body['file'], id)
    return JsonResponse(data, safe=False)

def unregister(request):
    dtct = Detector()
    id = request.GET["id"]
    dtct.unregister(id)
    return JsonResponse({'status':0 }, safe=False)
