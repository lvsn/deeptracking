"""
Simple logging utility for Slack API
Make sure to set SLACK_API_TOKEN env variable as the api token and
your slack username (for private post) in SLACK_USER
"""

from logging import getLoggerClass, addLevelName, NOTSET
import logging
import os
from slackclient import SlackClient

# level just above info
SLACK = 21


class SlackLogger(getLoggerClass()):
    def __init__(self, name, level=NOTSET):
        super().__init__(name, level)
        addLevelName(SLACK, "SLACK")
        self.sc = None
        try:
            slack_token = os.environ["SLACK_API_TOKEN"]
            self.sc = SlackClient(slack_token)
            self.channel = "@{}".format(os.environ["SLACK_USER"])
        except KeyError:
            logging.warning("Slack API error, check 'SLACK_API_TOKEN' or 'SLACK_USER' env variables")

    def publish_to_slack(self, message):
        if self.sc is not None:
            self.sc.api_call(
                "chat.postMessage",
                channel=self.channel,
                text="[{}]:{}".format(self.name, message)
            )

    def slack(self, msg, *args, **kwargs):
        if self.isEnabledFor(SLACK):
            self.publish_to_slack(msg)
            self._log(SLACK, msg, args, **kwargs)