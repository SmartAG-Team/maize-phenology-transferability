"""First-citation figure placement shared by document writers and verification."""
import re
import html
from reportlab.platypus import Flowable,Image,Paragraph

FIGURE_REFERENCE=re.compile(r'\bFig\.\s*(S?)(\d+)(?:[a-f])?\b',re.IGNORECASE)


def cited_figures(paragraph):
    return ['fig'+('s' if m.group(1) else '')+str(int(m.group(2))) for m in FIGURE_REFERENCE.finditer(paragraph)]


def figure_placements(content,allowed):
    allowed=list(allowed);seen=set();placements={}
    for section,(_,paragraphs) in enumerate(content['sections']):
        for index,paragraph in enumerate(paragraphs):
            figures=[]
            for figure in cited_figures(paragraph):
                if figure in allowed and figure not in seen:
                    seen.add(figure);figures.append(figure)
            if figures:placements[(section,index)]=figures
    missing=set(allowed)-seen
    if missing:raise ValueError('Uncited figures: '+', '.join(sorted(missing)))
    order=[figure for figures in placements.values() for figure in figures]
    if order!=allowed:raise ValueError('Figure numbering differs from first-citation order')
    return placements


class InlineFigure(Flowable):
    """A native19-cm figure/caption pair centred across the body margins."""
    def __init__(self,path,caption,style):
        super().__init__();self.native_width=19*72/2.54
        self.image=Image(str(path));ratio=self.image.imageHeight/self.image.imageWidth
        self.image.drawHeight=ratio*self.native_width;self.image.drawWidth=self.native_width
        self.caption=Paragraph(html.escape(caption),style);self.spaceBefore=6;self.spaceAfter=6

    def wrap(self,available_width,available_height):
        _,caption_height=self.caption.wrap(self.native_width,available_height)
        self.caption_height=caption_height;self.width=available_width
        self.height=self.image.drawHeight+8+caption_height
        return self.width,self.height

    def draw(self):
        left=(self.width-self.native_width)/2
        self.image.drawOn(self.canv,left,self.caption_height+8)
        self.caption.drawOn(self.canv,left,0)
