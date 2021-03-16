# _*_ coding:utf-8 _*_
import xml.dom.minidom


class XmlWriter(object):
    def __init__(self, doc_path):
        self.doc_path = doc_path
        self.xml_doc = xml.dom.minidom.Document()
        self.root_node = None

    def add_root(self, root_name):
        self.root_node = self.xml_doc.createElement(root_name)
        self.xml_doc.appendChild(self.root_node)

    def add_element(self, element, value, attribute=None):
        node = self.xml_doc.createElement(element)
        value = self.xml_doc.createTextNode(value)
        node.appendChild(value)
        self.root_node.appendChild(node)
        if attribute is not None:
            node.setAttribute("name", attribute)

    def write_xml(self):
        with open(self.doc_path, 'w') as f:
            self.xml_doc.writexml(f, indent='', addindent='\t', newl='\n', encoding='utf-8')


if __name__ == "__main__":
    xml_writer = XmlWriter("xml_test.xml")
    xml_writer.add_root("param")
    xml_writer.add_element("node1", "1.5", attribute="my_node")
    xml_writer.write_xml()