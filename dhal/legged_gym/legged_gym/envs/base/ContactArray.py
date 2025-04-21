import rospy
from std_msgs.msg import Header
from std_msgs.msg import Bool, Int32
from typing import List

class Contact:
    def __init__(self, id: int = 0, indicator: bool = False):
        self.id = Int32(id)
        self.indicator = Bool(indicator)

    def __repr__(self):
        return f"Contact(id={self.id.data}, indicator={self.indicator.data})"

class ContactArray:
    def __init__(self, header: Header = None, contacts: List[Contact] = None):
        self.header = header if header else Header()
        self.contacts = contacts if contacts else []

    def __repr__(self):
        return f"ContactArray(header={self.header}, contacts={self.contacts})"