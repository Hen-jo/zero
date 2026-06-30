content = open('d:/paper/drzero/paper/gen_drawio_content.xml', encoding='utf-8').read()
with open('d:/paper/drzero/paper/figure_architecture.drawio', 'w', encoding='utf-8') as f:
    f.write(content)
print("Done")
