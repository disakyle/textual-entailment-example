# -*- coding: utf-8 -*-
import json
import traceback
import re
import signal
import time
import boto3

client = boto3.client('runtime.sagemaker')

def lambda_handler(event, context):

    def run_prediction(input_json):
        response = client.invoke_endpoint(
            EndpointName='textual-entailment-endpoint',
            Body=input_json,
            ContentType='application/json',
            Accept='Accept'
            )
        result = response['Body'].read()
        return result.decode('utf-8')

    method = event.get('httpMethod',{}) 

    indexPage="""
    <html>
    <head>
    <meta charset="utf-8">
    <meta content="width=device-width,initial-scale=1,minimal-ui" name="viewport">
    <link rel="stylesheet" href="https://unpkg.com/vue-material@beta/dist/vue-material.min.css">
    <link rel="stylesheet" href="https://unpkg.com/vue-material@beta/dist/theme/default.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.32.0/codemirror.min.css" />
  </head>
    <body>
         <h1>Writing A Hypothesis</h1>
        <div id="app">
            <md-tabs>
                <md-tab v-for="task in tasks" :key=task.name v-bind:md-label=task.name+task.status>
                    <doctest-activity v-bind:layout-things=task.layoutItems v-bind:task-name=task.name  @taskhandler="toggleTaskStatus"/>
                </md-tab>
            </md-tabs>
            </div>
        </div>
    </body> 
    <script src="https://unpkg.com/vue"></script>
    <script src="https://unpkg.com/vue-material@beta"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.32.0/codemirror.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.48.4/mode/python/python.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/vue-codemirror@4.0.6/dist/vue-codemirror.min.js"></script>
    <script>
    Vue.use(VueMaterial.default)
    Vue.use(window.VueCodemirror)
    
    Vue.component('doctest-activity', {
        props: ['layoutThings', 'taskName'],
        data: function () {
            return {
            prediction:"",
            layoutItems: this.layoutThings,
            taskCurrent: this.taskName,
        }
        },
        methods: {
            postContents: function () {
            // comment: leaving the gatewayUrl empty - API will post back to itself
            const gatewayUrl = '';
            fetch(gatewayUrl, {
        method: "POST",
        headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
        },
        body: JSON.stringify({hypothesis:{0:this.layoutItems[0].vModel},premise:{0:this.layoutItems[1].vModel},task:{0:this.layoutItems[2].vModel}})
        }).then(response => {
            return response.json()
        }).then(data => {
            this.prediction = JSON.parse(JSON.stringify(data))
            return this.$emit('taskhandler',{data, taskName:this.taskName})
            })
         }
        },
        template: 
        `<div class="md-layout  md-gutter">
            <div id="cardGroupCreator" class="md-layout-item md-size-50">
              <md-card>
                    <md-card-header>
                        <md-card-header-text>
                            <div class="md-title">{{layoutItems[1].header}}</div>
                            <div class="md-subhead">{{layoutItems[1].subHeader}}</div>
                        </md-card-header-text>
                    </md-card-header>
                    <md-card-content>
                        <md-field v-if="taskCurrent !== 'Free Practice'">
                            <md-textarea v-model="layoutItems[1].vModel" readonly></md-textarea>
                        </md-field>
                        <md-field v-else>
                            <md-textarea v-model="layoutItems[1].vModel"></md-textarea>
                        </md-field>
                    </md-card-content>
                </md-card>
                <md-card>
                    <md-card-header>
                        <md-card-header-text>
                            <div class="md-title">{{layoutItems[0].header}}</div>
                            <div class="md-subhead">{{layoutItems[0].subHeader}}</div>
                        </md-card-header-text>
                            <md-card-media>
                                <md-button class="md-raised md-primary" v-on:click="postContents">Submit</md-button>
                            </md-card-media>
                    </md-card-header>
                    <md-card-content>
                        <md-field>
                            <!--<codemirror class="editableTextarea" v-model="layoutItems[0].vModel" :options="cmOptions"></codemirror>-->
                            <md-textarea v-model="layoutItems[0].vModel" autofocus></md-textarea>
                        </md-field>
                    </md-card-content>
                </md-card>
            </div>
            <div id="cardGroupPreview" class="md-layout-item md-size-50">
                <md-card>
                    <md-card-header>
                        <md-card-header-text>
                            <div class="md-title">{{layoutItems[2].header}}</div>
                            <div v-if="prediction.successStatus === 'Success'" class="md-subhead" style="color:green">Congradulation, you have cleared the path.</div>
                            <div v-else-if="prediction.successStatus === 'Fail'"class="md-subhead" style="color:red">Sorry, please try again!</div>
                            <div v-else class="md-subhead">{{layoutItems[2].subHeader}} {{taskCurrent}}</div>
                        </md-card-header-text>
                    </md-card-header>
                    <md-card-content>
                        <md-field>
                            <md-textarea v-model="prediction.jsonFeedback" readonly></md-textarea>
                        </md-field>
                    </md-card-content>
                </md-card>
            </div>
        </div>
        `
    })
    
    new Vue({
        el: '#app',
        data: function () {
            return {
            tasks:[
                {name:"Entailment", layoutItems: [
                {header:"Your Hypothesis", subHeader:'Enter your hypothesis below', vModel:"Type here."},
                {header:"The Premise", subHeader:'', vModel:"A soccer game with multiple males playing."},
                {header:"Output", subHeader:'To clear this path, predicted label should be ', vModel:"Entailment"}
                ], status:" 🔴"},
                {name:"Contradiction", layoutItems: [
                {header:"Your Hypothesis", subHeader:'Enter your hypothesis below', vModel:"Type here."},
                {header:"The Premise", subHeader:'', vModel:"A man inspects the uniform of a figure in some East Asian country."},
                {header:"Output", subHeader:'To clear this path, predicted label should be ', vModel:"Contradiction"}
                ], status:" 🔴"},
                {name:"Neutral", layoutItems: [
                {header:"Your Hypothesis", subHeader:'Enter your hypothesis below', vModel:"Type here."},
                {header:"The Premise", subHeader:'', vModel:"A few people in a restaurant setting, one of them is drinking orange juice."},
                {header:"Output", subHeader:'To clear this path, predicted label should be ', vModel:"Neutral"}
                ], status:" 🔴"},
                {name:"Free Practice", layoutItems: [
                {header:"Your Hypothesis", subHeader:'Enter your hypothesis below', vModel:"Type here."},
                {header:"Your Premise", subHeader:'Enter your own premise below', vModel:"Type "},
                {header:"Output", subHeader:'Enjoy the ', vModel:""}
                ], status:" 🔴"}
            ]
        }
        },
         methods: {
            toggleTaskStatus (response) {
                const {data, taskName} = response
                if (data.jsonFeedback) {
                    const searchText = data.jsonFeedback
                    searchText.search(taskName) !== -1 ?
                        this.tasks.find(item => item.name === taskName).status = " ✅ "
                        :
                    this.tasks.find(item => item.name === taskName).status = " 🤨"
                }
                if (taskName === "Free Practice") {
                    this.tasks.find(item => item.name === taskName).status = " ✅ "
                }
            }
        }
      })
    </script>
    <style lang="scss" scoped>
    .md-card {
        width: 90%;
        margin: 20px;
        display: inline-block;
        vertical-align: top;
        min-height:200px
    }
    .md-card-content {
        padding-bottom: 16px !important;
    }
    button {
        display:block;
        margin: 20px 60px 20px 60px;
        width:200px !important;
    }
    #cardGroupCreator {
        display:flex;
        flex-direction:column;
        vertical-align: top;
        padding-top: 0px
    }
    #cardGroupPreview .md-card {
        width: 90%;
    }
    #cardGroupPreview{
        padding-top: 0px;
        position: static
    }
    #cardGroupPreview .md-tab{
        height:100%;
        vertical-align: top;
    }
    textarea {
        font-size: 1rem !important;
        min-height: 175px !important
    }
    .md-tabs{
        width:100%;
    }
    .md-tab{
        overflow-x: auto;
    }
    .md-tab::-webkit-scrollbar {
    width: 0px;
    }
    html {
        width:95%;
        margin:auto;
        mix-blend-mode: darken
    }
    h1{
        padding:20px;
        margin:auto;
        text-align: center
    }
    .md-content{
        min-height:300px
    }
    .md-tabs-container, .md-tabs-container .md-tab textarea, .md-tabs-content{
        height:100% !important
    }
    .md-field{
        margin:0px;
        padding:0px
    }
    .md-tabs-navigation{
        justify-content:center !important
    }
    .md-card-media{
        width:400px !important
    }
    .md-button{
        margin:10px !important
    }
    .cm-s-default{
        height:100%
    }
    .md-card-header{
        padding:0 16px 16px 16px
    }
    </style>
    </html>
    """

    if method == 'GET':
        return {
            "statusCode": 200,
            "headers": {
            'Content-Type': 'text/html',
            },
            "body": indexPage
        }

    if method == 'POST':
        bodyContent = event.get('body',{}) 
        parsedBodyContent = json.loads(bodyContent)
        taskHypothesis = parsedBodyContent["hypothesis"]["0"]
        taskPremise = parsedBodyContent["premise"]["0"]
        thisTask = parsedBodyContent["task"]["0"]

        timeout = False
        # handler function that tell the signal module to execute
        # our own function when SIGALRM signal received.
        def timeout_handler(num, stack):
            print("Received SIGALRM")
            raise Exception("processTooLong")

        # register this with the SIGALRM signal    
        signal.signal(signal.SIGALRM, timeout_handler)
        
        # signal.alarm(10) tells the OS to send a SIGALRM after 10 seconds from this point onwards.
        signal.alarm(10)

        input_text = {'sentence1': taskPremise, 'sentence2': taskHypothesis}
        json_input = json.dumps(input_text)

        try:
            jsonResponse = run_prediction(json_input)
        except Exception as ex:
            if "processTooLong" in ex:
                timeout = True
                print("Processing Timeout!")
        finally:
            signal.alarm(0)
            
        success_status = "Fail"
        if re.search(thisTask, jsonResponse):
            success_status = "Success"

        return {
        "statusCode": 200,
        "headers": {
        "Content-Type": "application/json",
        },
        "body": json.dumps({
            "jsonFeedback": jsonResponse,
            "successStatus": success_status
            })
        }


