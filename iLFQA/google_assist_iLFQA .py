import os
import json
from flask import Flask, request, send_file, make_response
import sys
import time
from eli5_utils import qa_s2s_generate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import iTA


app = Flask(__name__)

def processRequest(req):

    answer, timeDict, context = ita.get_response(req)
    return answer

@app.route('/static_reply', methods=['POST'])
def static_reply():
    
    req = request.get_json(silent=True, force=True)
    resu = req.get("session")
    result = resu.get("id")
    Qi = resu.get("params")
    Q = Qi.get("user_question")
    val = processRequest(str(Q))
    my_result = {
  "session": {
    "id": result,
    "params": {}
  },
  "prompt": {
    "override": False,
    "firstSimple": {
      "speech": val,
      "text": val
    }
  },
  "scene": {
    "name": "question",
    "slots": {},
    "next": {
      "name": "actions.scene.END_CONVERSATION"
    }
  }
}
    res = json.dumps(my_result)
    r = make_response(res)

    return r

if __name__ == '__main__':
    
    ita = iTA.Loading_Model()
    app.run(debug=True)



