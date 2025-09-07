import streamlit as st
import spacy
from spacy import displacy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
from wordcloud import WordCloud
import json
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup

# Set page configuration
st.set_page_config(
    page_title="NLP Text Analyzer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load spaCy model
@st.cache_resource
def load_model():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.error("Please download the English model by running: python -m spacy download en_core_web_sm")
        st.stop()
    return nlp

nlp = load_model()

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 50px;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .highlight {
        background-color: #f7f9fc;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        font-weight: bold;
    }
    .summary-box {
        background-color: #e6f7ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1890ff;
    }
    .entity-box {
        background-color: #f6ffed;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #52c41a;
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: gray;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<h1 class="main-header">📝 TextInsight: NER & Summarization Tool</h1>', unsafe_allow_html=True)

# Sidebar for options
with st.sidebar:
    st.image("data:image/webp;base64,UklGRjIfAABXRUJQVlA4ICYfAAAwigCdASqRAcgAPp1InkqlpCKhp5QboLATiU3cLR4hA3f8z2Msn+Vf139v9Iayf4X+4/szh8K28p/nnz7f8b1OfnL2Av0i/0f9F61nmv/nn/O9T7/X/uT8AP8d6pn+R/1XXN+iF5d/s4/tJ6NP////+uTei/7Z6T/HD8d+Xf9t7VL07+3/t5zV+rvFd91P3P969u/9n/xvyS85/lBqBfjX8e/w33DcPiAD8n/rf/W9NT7HzT/lv9H7AHBlUCP6P/kP2V93//T8lX15+1/wH/z7+3dco0FC7iAoFFUOt8lFggyCK9gKKHOy9nDOFnW45HwiOp3dYcF2NHL1nh97S6ktMfWmke9acRK9gEuTcQEW2v7P1dQ48lkrPyFV5qHVrhDlP0EVBHPAiZ/6NWV/M18AE1mnBwIiUlilwL3CZRPaWSCo5f7RQPN8z5HYBFMNp2sBJbl/EGQQmM+BXeVVgOH+ootdhpionD+z46CxmfhOo4N+teiRJvy4BWlxTQt5ZejxZNAU59Hnl9X/N/Cg3/3bTKf8qKT7NfA14bKwJmJXsAloSjGJuiAhtEcluKoiEZ5XXp0Gtxcdk/5wlevx/hhuGD1XCh8mtfs3jNjcUOnFBHAc0dSGt3763yUWCDH576/o0Ayx6DbLqglGwb/IKHPg2Ivf5Gihmf/9vWEc5ngoEievXWtwadpdEBGOfVveAYKT95D7mfssOUW7WKCJ9+7LVdrHS3tKWUByrv5Exq/RbIqtsh6vQ9zOGW3amNsL6bxekX0qMsa33m2kRxSLe8fl2yGfoK+b8rmq/FX/Ox8HlUQzyS8tt0XCf+Y+3YifoXM8B/WK1+SPy831k3/inAOa/0j2GpY16UL1BQGFB7TIG5u/+oZyp8zK1S4QLVllkHPjLgYns9NcfScmmE16p+ZEuzmEIM6emf/uz99wTDeBAfX/8QmURovyrdM+l00wt/1//FP+KfD/ToloPlJxPXLJpUboCFKTTqUyknIUvkUTql9mQfM+agOpAeKtOlTY/EhC5rTxNcA2rCexBfF5C1SNy/NFh4PAunmMPijxbY9KnCm3XN+x88Pe1ZtPNwUwcsmY/FSWFdFPNOiEfSx10Eb5OQSbTMqi9k/oepKt/q1SHlfecerfHlgKZJ0Jv38en9ChIqmJYfQqt1rgukqGpbOimj98gr1zT0F/DzIHaTNkRzgVLpdKs3tRAtOZ8UOSGhTp5xlmgKodgosgFK2VKZoD4Cn6pckea83AEy5Vq/yG+e8kvDk5uJK3xiHqM/cFXsknF+u6Lx6eXPYgl/npWMn35/zetQE8708kqojY9q+4B/VLQr+/uL6kt0N3KQogM9mS2CtFgaWJw4XZvylnZ0YCwVKnzy2TVdoNnrbE+l0/HqKDar9L4wcs6e9MfD+9ijsEwlBh1PGFf3QVeyX58nLf7gqhQ6s4uJF3fl82jkyBVDrfJRYIMgis3Er2KUa3fvrfJRYIMgivYCegAP79rWDf4iQwnuEobOKEPOAQQxu3jc4gyGsANzRhAauelQ2zb2Y3aCF0LJOF4EQZhTrw1m+lgfWwJGpGwVsPakMjSuzd/+Qy4YbdJ17EcfPkBFn195H4wxxbX4Qa4iYUFvKFPtz/FDQZPPnfvgdpt5+yDR8PMi3H6xBvnHs/Ebc8RzsJeDS7AJY81MY2PouoQILFECtVKsInmICOBObfFx7kb9yj3lC7DURZj2AlN3ex4vLbgE6r9WMFWLvJhOKix0l0D8DfrmL75LpxzxssffcEl5ELnhWhSTLzle0gz+30rupZnIRkts2xEkuCtguksTu++OnvZukXK83j2B8kTIU/P/qxbWi53I3ZIB0sRJocO4+jc3WfuWaTC8I2HVSU8xqWGpvaAT4lEte95ld0AY0Zqpf8+25AKO7BL+mTUin6RRR/jlp7ZtsVqTSCTeICaB1PiK+y74av9ufCoTg5KW/PQb3pq617qYZQbEla992Wd3tPVCcNYSA8BzbDfM2p0JFC04L32hxY6zh9kuFAkR9fr6wgHM7zbZ9yIjGjE0sMjhudPNngyRzcKf41axMjRwrpNn4wsstPhbev2hITuvyqV8cRwRc3PKnme1qBpdTYEB/3PXYlWn40DNwZLdivE5V2KO+YH5zT+PCHWIT98dQF/3/QHbFyIANxHXynYb8FKpTEydFRQhQRjPmorFPP/ZW5Amo7jdPyUg3mQfUj9UNZGxAJhA2KgcQzjkKaQD8VfH4YWBTBhFeWe9uy81hKlOXnb2+nBiUEKsTWQb6XQ81nJUV1n6l/rxPvtrYEa8/t1OdEawniT+KNLLFsBak97Jc8T7XFwHbz3L9TKtpgOxc/d06scD5qtzpjrOys4dQYhyVl6LJtYeZAHplu5BOwwxQKfoSTsBaRvwXm2JjppHrSu8GdhXxKJG8b5tWf/uqtFYrxbpQapLxzXJo6axUwSmqb312aD54fYrn8+3ae/JF+oRk5R+RMSue/JaUFclW0Q1BxPbsJboEHy/Z1RUfX3scnjxfqfpgu41Ap1FMul93c17seVHCjlIFd4pMZFftgQigHAut28Bro8JYhkpho6KJky3ywOr/83OXLf7G+P5BB+WOfHosEIAKUWlq7ouLBpDXIMLHJEqJN3qgPJ5CH2FfrivbpHtP/1CcgMWpqbpfT4mzRGiYkIITqn0kUGo4/FUJ1lz6ZgCVwZNVdxS5DwXSVoFPtiU6gGK2fZlEleUpieFoOjFK6WkcM7y9EC29upqcilOXCDbOAahB6f5+g7dItuVe/aJjt99UyOQ+A/gKqnUZ9csoBUhRA1fgsH/HUvHUJlWS9HVdNuWfGKgpz/f0KxEy3Jn8gu3wT4MG1DaOyXibNCVekDemvvOKomrE5ivms9T5nYWgTxzb2tXzyrHLGeZiSJJga9AAcP5wkqo2SlQj5gW9S7jV3NhVLu+q642+UG8GnMTk4L+UdTPkqgrLFPHO+CSiE5D9bd76serUm+lyDP2YJCsfesXY8DDdRSi6mp4YnkB+PMM782FVn5ayhYaSN4P6I3I8EiCsue5LCDu8hMDnOqA+sSa/a2MtoXJWC5Snsb1ZzWfD8p3wU5Zm2GGSoSfvcg4xAjRBa86tlH+Ym+rpegJBCQSgAd94DPn85IW9lAVMo73w/GpnrrjipFdPu+iWyPVdA3LBS4YSq2blquw2m6Jf+GAw/E5IpctoL94guvY9SYe45TNsrr56c/CYDAfE065ZTIf9BcAzVlkTtHSkJQ5naSjYFFNoeTYEJBEDZleLIn7rDdPfmV49Y5fN3Mz/kebTfmprs7USCTHFNlwdufMNzTarS1EEh1Q3FyitvNnVoS2bPp93W4fEubP0Uo9QrLjHMhUj46OqFKKb+jOqJ7Kea+ttyZu4RAg0Mnrqw8JReCySAK4BPXREJ+sd3uNgCgN1achaBVuaYHPcekBmT/yDsAoD3wAzX8mcw/G0+e4jA+sY9EnoFf0hEx/0bjgzyouOWq38b43YKU/LzGD0BeDcz/uX+879+IYHrWuqRnL10XG3TU2FB6KGXLXoOrKogvqmWHxqE9lsSLSEwzucwocX2p2oWWEVy7ZdejUa/bTMJfdF77mWMioTdHqrWuM6QtSkHB9v18ezMPY2GDy01XTjw2sQ32l2ooucu5J6Fvs5Mw2veAfXDB2uPTHFiAkVBPDbPvIcG9mRDVGnHP3jk0O0JFc2y4r1ENIW04Kr0tYM14QepbtOdnCbz608AoFjoP09kDb07XsXnRYNMxuEhHknGOhWEHHUIdTN8zNwE30aXq2DHpznFdIyP7Ntp3VGho9hyAa385aETrH3Eea7nzjTuJwwgAyeRAjLocDz4L38ud+daX0XyCfK8LgglxdOTDRDYVO8hwhD+yE5/hILEA3nA7iRfE425BHyOT9+/Z7OzfJiw3REPYpy1/jiQ5EW15zQBb6o9GEFySBDgD8qgC8rpvxgQaYlMHC3WeGN1S3LAuSwt8UwUquyxQMLZ62cmoWKGsMfOcsKZGylvjgMh00N9iyf2faS5DdZVyN2uQO/qQq1z4kd0RQUreESQZiQ4ZyVJFj7YGQgujO28J4QURBINZ0w+gARKH6jP5FuJSuFE8+X4XXQxcat66crfg1MdO8BCZs9GrE9N/pXTLeCvVYh9Zo8ySb9atXIOgcf2Jx9gXOVsvL5wrea516XdT4w+F8DCKpAS8HEyHZO0JNUs0wVmpZzjxZqWD05Vy89zKCQlT4qZTTD5NHZ1w1iJWCbBXsNCdy1NSqduBRU9ei4GRKpihNslrCI58yBQ+31yb8WV05hOUYZf/zikrbKMBFuc8kDnpJvnJp4uTDyAGC6gMwqBLNLMWv+R5Zo/usJavNa+ry0FM1MqPEZJ++ZcgMRTv9ajReIQdKSy1TnD0ku5sUwlrnLEeIASao6AdIuxaT4h5qzNlf3W87j9UOxykxk9x2dtjPzZ0j0yd/SfN56mwEsFbohtmJ/0kzZi5aYPycNVi4vVLU39XRY0rBS7ERpAJH/W2FQ5VvaoCVk2lFg612tr8RJXYuswgFoBpDQs61VyHAmcCJX2gdmJQNvlwQ7PAD2iq0AIEvhzx0UKDWvawQpCWeNB+udrXlh5mx6t4lwLVBWJ/3S3EOPvOgAu8fFH3nBq+7akXNP+5mX9hZIGMwTcBc1RXQx8LR/aRsebWO8NIqfwpK39FN0WEG2+jmyMKWmkWGBOTu2X6jzNknY/AcxraDDWKSnZpKOLZD+zU0rn+/lMMRsKFm7AEac78SFmHKf7Zm0J/g+eIBTP+s49rUXdWJyicvucdRvv/1HUA3OOOEh2R+6BtljmLUorryz8i74+91A/A82TZwSS+ncoDT+BWBdgYL4IRK1YpfIk/DFF7Ii0L85vJPe8f/Du7fatNW7QUv4VL6nXUe+Smk5bFDPGhxuGfBZo54nYHOFoiybECWKnq/Lx3bVmbdFBwcjWc8hXA+jxQMSVuwBq0gqfo6VFAkU3AUBDuebGDuphwCbkTXrxguV1fuoD3/GzoebLruo3/fxtMG9zP1RKupDL/Vz5uVcUAAAnsvWoe4LmuOf/4N1LxhVg7y36VX7LJs80jGbxa5al2OAE7vJwT+HjP4LMRxKB7QqcLIH9AtU8oK5jG3rpgOz62QjfagZe9PYSrhbX65hBPz2PU4MDxOXnEXVH00UKhhmLwDIxufClqbyu4Spam1N0s+YKYC97cRqseI9lWaG8FH0u6dgXHPfadS6fNZYvOrcf5ycm3/APEJT+BeY+pd7K9A+AklNor5pNcytTbaop1cLIAPhn3XuShd/Wa3CBhYG3yoTm0BdktTyVMonZVcOdMOK2tUK3A8eBuP2LaPDzj4bfGFq0gA7LOBih9QTESYcrEzJo69n4va/lFQZXHizErlXQ9Ub1abeKozWwL+2lsXcHdJ/Q0TfaCxdjdVXYEIWMvtCwj4h2z/wO/1wrIM5H5BJG3tv9QPVRxjmgxeVKDBuRrjnJ+dYZnQttktEVlREZe2zT2cI1OgxY717vXc8C+gJRQzL9yaazu47nnZQYd5Y/KGKO+HZjmQI0lm1EFqyy2moWnCShqOsbMoRRHEwqhUqJcHOsk2jxPrP+GrsADRYf86KPhC7Q/jsPDvuCYwfZFvEul/kBPy8nliRFNTPJQ5BZz5AWy3qj7uvql4OsbM+sE38bKBfus+oZVG/FqgxL7/GCH4GLGriSyFN2Q7186cfEkBz9rfM2Qp87hkfE8HArCklrVmeJ/hjyyocyRIl0VRwaGDddN0QdSbXQR8cpa2aIktuNXbuTO0epLn7HYHbwdAePDl2X/AsjhJ9SuRE0WjbXrlHnmkSLGcU4w13ZZsmjGc6Pmn1c0jye5wgesuhVTwvh0uKc1vEGskOjr7iiDriDRqxvtSUargnRpxJvls4D7Yays2MKYgqctmhtAxCm77sEP1k2LFb9K/ugmx5bz+4XwCjRF7QwSkRG3dCf8gBUYzZz8NX3eVuH/Ty2TGcJOqJ06e9GzeUnaOWpcHrLz+ot06SAdOBpEw9BfdWoBuPkQAyCWd3zsHEXMSnp+rHF8H8WX5x8pXU73JnUclYW3O1XjkS+pAwr25cVLuO+GSfEli3W9ZUUt2DAm1XyWNJWHkeAejbN+KvuJQbr9fAXAtjlCKdFPRlLeDtVe1XQUJu5Z8N0FQ36+uHED6vEBzoKy4I9il2vb9u/uur0Q59NMnUSviCjigUr6QAk8AyjsEN0b+yU0bNYHawANjfWlxUWApbc5uG6mpAZ1+klYBkjkrK6EvKwWYiOucrRn5zgkVFjaZ8tFX5tILs+5RGOuVeZdj+pp5WTemTLP0UwU6jzLaTUy0BLdVdULfMItshl16OzUb/gtGLFB7AsOVCqUjmq+3Bk+Th5Xi/5VaYhFfXLfV6xDMFup1qHp81IUJmwpIt9q0DcsOEsSUW3nGFqeyMMdLbU4B6gA9EEpeeAlTzNp5Q7UfO+B/JU+sGvU8SCu6zimfPJGhlDrrjmAUOGzRe6Gnj/nDls8ENTNMhupbjlm/mxXtKV6cy2ql2CzENwPEAON+aKr3LVLVLE1dwFVJCfL6kBUj7PbXNSIwllfS+eGwHrVHPIUhPCZwrAu1Xt9Mrc8W9H0/bw4NZo7s9QhwcTkmnZ2WlmqRARQCVsbrN+SqGW8wJeFBa0Zyx00I0Vc9+fOwFl4fFoDiUuzsMSzL75h2SOcX+uGj92ckzrhzu95tF3KUjw+g/R5kf3FoZgZlsXDfQlWtld5AtEy1FAXzRpFhw+xSl+jIXnaFWH07awraC0wYNY0/Qlsd/q+fF7uw2dbIAtw1r7dNeyqtEeQT/zRznl0l/e2pD+UxzvQmvbYrlkXfk1XIAM62Z1mBDFZ0avOUdnrgC5cQ2+yG42J29eTXuE++lcjEmHvajgPgx19/4faXzX/QD/PdRXdUhQ5wfS+WXRCaOWwGOFOCosRTX4hO1xKJz5pix87J/9wuVfxF48arUoiC/ZtiUA7eBr07b+2Kmi2g2ZYDXwCnWY4HSZgDzmSOAbxw4APK9k09puoi5QFDtJD9Xr2NMs9Uzo5C2iSaU5w/BrNjNGjuiEFHlog9oXnMmMDSQ5EjV4ggsOUN//3g/hN0ZMgbTWYxUfQjMSYw7FHKyMDvwAe3Igx15CSCufow7D1ipj9mQ1u2Smeb8oh9b2JG99wdADxaXC+044rPWElxF4N5IAsvzMsrLYZJgHj0mJZAy/pOqWyqlTBxSIjFnxzc4iE4i9bktt+GBx4m4HkTgV+HUfLZmcLpAwss5JOT76eYi8n/jZwP8EhPBUWMd7hnP5K2QLk0pT4dhUsb0OzGSSrXDZqiolxl0QP5y6ow316gDCzwDwhHjZgFAE6EBnbZ1AZpsPctaK8ELbqOQla+bqcKWJ4l3VfL/pxrsQz2mUOJEYyk9Y96TaBrak1GI3h+gSOk7dvb2sZMwfw3rVITrUkqG5bPqWgN39ArLCJhdDwbIDMMXR16vgDwiIdw3V1jLOPLxhCj7y4ONqC5O6aiHMp+G+II8TrkPJD8KiWyLYGzVA5RCgaL3GWKIZFbQZRvhiOBJumltvW/N919B+ENYNyVztfh0+J9rkXdM+n9S53saW1eVIA24LLx8oWMIpw5bKnIGFPtZWxQfBptFnK+pK1tfkFnql7O21bG0FiVloX4PxUQ2rbO41plyIV9OJ+O8cb11rZ7HuI6lntyl3k886VRjYhUv02qR0IRUArdvWt7mwYabHadtdAaz8dpIPKpkLc3C6e0nFVwHL8KhPhoeC8iQKfIT/gBgGD+xGaDyu/qjx9+7bjoyuvl7sbRrxQOa8punqmvIt/+4s1JfGohdurJFWM7SJGNHzxL5LkCFRlEd6UUkeOxamY4l+g52J3pfsk37UYpcuycO5m+QHlUIcJcRt1LbifspCXf3IuVgo/ClLMt9clRfMSJL+zXx2QeRMlmFaVq7yGMxWbvCBg42wQoDzCnAFy9JuIaCL+LNI5kZm8dNxFgCT4fN4GCmiH80bLla+QJlEvQHjWZb7ifcjcnXATcMXSqmIsOtaaF52g/9mi7Y3KE1kus2lmqarGqU9gPTOp+ht6BXiZVQb4DIaRzILvPzBHe9Dg4Z4bl8ekTVzhVa5ha9WEv7jQrWlC1qFq7oEn4E8LU0PCSD76nqeD+2a2/EjoIsMSa7Q2Wztq/0ZoB9ufZs0YjP72FajBCYpZDmHMsuJBGT4fIOHiM/hiDLVLgbPXkR61aYEJZOjQi2sAgfNtj1LutfWErlLDjlv2x4JXA996c3RsWsXiQczvjh6bmy0afFt13IqvjbdDjm4S/vkKGvtHYDbml8l9ojTA57Ys5onMJd3wCc7g1zs6KtPIddnUYI8YWY9g6WtWm2oq71Xs9WnSkforh6x/6h+2tpix2f2SzEd964P/xH4/QN/+i5SifvJNIiLvwfARYfSQ7nVOE7rORKKACTlb5ab0GEXlvO0s8xLhuKnEAOe9lqk3jJ8+9JvxrY6vIYTmCD1Pp+b8c/k5KSEsl8A0erYQUWo7aSPsHkkT0N8wok1/TFqFOvPUtk4u0HgYhpv6ufDwZ/E2jxXYrEMep1nuQVd/i8gqafGQnWgLEKg2g7gubhDesjXR1bn/JiA3oltdoTsFCPIjvysV57Mwtoo76EeobY90Xb8G1BdBbqBhTQ+LgSNdJ7nrG+44HIpXAmB9ykEmtFl19n2LobVWRv00qmqgxY8zuGp4Vo8refbCRoLzGV0VOAma7CRgVDmdmXfmVO2visdIjQ/7gHePaCA7XmY7NKv8s5wDamI0KD22HChBKohi4ykHktKWRRpm6XHxKmulEKEEk730xRuqLjT5kOdOlyNHAGvS6NpmJ6SvMbtIBRn9WxvswuBoh9v7o777KuO1B7tX597Ep1i0jcd/gs5+R/sdUUQvCfJE1kP4+Ou5yN1DsQjWT6DMlhunoNWF7v8z4c/KqhcmnXPdozgziATFnB/QlO57lNrrw73ddahGaYMEmO404MWHkFPDc7ao+/bqCb1B4KYBtSPGULn+pEuZ6NOXC7vn92mbXixSOAaYMF9UwO/+oPELDq+K2di+9xNRCiDgB154/h51wz/cjv669LhenbL3a0v/QKnegz9QhPWvuGO1RDOpKPYmvW5jT+PeZTysPRv0/uw4TVEmwKpr9zWnTGRMlZxmc9+b8Ka9wZu4lOfFdUU7nnkhztoPf8kFScYP2WoYHrj65+ipuVAnUPW9N3eNW3CtVP7WSR/o6da1IIJNHVtfKPto+aCIRUCSuW4BhiCevfa0ZolHNCw+/iZiVXVMD8IS5Ienyv4DB0TAuv7zy675G40X2XxJEK2/Hcx4NXFywwlKRr5fcsWYj8OOKNbiajMB79OzIgE8n/fvY6C7AJonAhGg4z4ltYPoVu+4X3HrwJtIXo7KPZhrrSBOw5tUH6n/Z4sjlSapFeJata51NKeZqfM8/bbHg/to67xfGSI6UZX7KkC8R4C5m/Ip+pKue1zvghwwq8RhfXiW0bhP6Em9o4lCw9On9M1LS3iDR/jyozq6QISzYhkMA8W7D18ovWS7YcZ4RHh+2anVAHiyJcAUcvNHZ8HbxzRud+XoW8T9XU5fOMVc9QsIcPTeJAhicJN0NKT+Cyto2Vzoo6IOP8Lvf7izF028IQLdXyD/dVEGuwU5g+BdV5091gaL3JSLSwupFdY4hAroZLvm1VPw+EgIRwPmi9/16pyL8eI7M6fSUfKR1CD1X2muS8ZG3AYZHy/jCnzLrXV3DVf/xGu5vMf258A5AmozCBJuDUy2obV9yad1mrx1uDwgjwYyGGs18uFDM5+7/wjJALMm6ZvmG047Z764DBrdbQAGBe1T+oCGWr+UAtzyvo47xOLsUE9AW3M2xkpPzxlYmcCl5ss0DH9bdl/AYCqHo6i+DZRasrZIRSRjsEU8dnF2KwoVGcXsK5blRkWpeRckyJwhE2x4qJ8Yay/Y0jnUbWNHK9OFe13ZEPJB7ETr2pMnQwFW+INPGmqAC2oRYDRsubX3wplQ1hab9EP+EanDCL+XxRfngggxnXotM8nzcdiPH+BYde1E21thS+4aRRGuX4UtPxDQTt6TXK3Zkm0hdgX3oA65YIH1JkFpYj14V65uTxNBdmIFU6JdnIgL/1fA03jqFYR+immG4ZRTteXU9m37KDuRfBLfiQ5CDnfRRRXyEOlwxlo7vkxsX/I2P5jUDrwtnpiYOHE8m+J/3zhFi6abiEZDethsKCoipdf2ZuMQOAYTXnGn9sUm+9Z+IHSvgoBMQFhUjS0NVaG0H4sUnjsFyq6hhuLJyskv3h+H1BAlatcqnKjOjeBYETE3VNavEBnb8F0y+w/C3Co6rroySl4lXBUWeI5WlPkGrk5qPTprn64RbOIQ/EtRBxT+/t32kYAaQjXIhgj/oJXMz3J4g0+amSo2mxNBuAsN4H1W6aSTtI7JA9PX0WNxvEBVggWRiaIGeml3adunmyKkQb1AB5ANKGpZy/EAYpdra2bqJ8SnehFkwtciHLpvpsZ2JApjY+uNPPfl/eF1iwG+F8h/CVQyunZWB9M+R8U0dMxFQOMpFZgX9PuGbqDiAr7j1MEFIgXgOkPyyUnqn4kJ+3gV3quo4AAKS68/I6W3oAAAAAAAAA=", width=200)
    st.title("Options")
    analysis_type = st.radio("Choose Analysis Type:", 
                            ["Text Summarization", "Named Entity Recognition", "Both"])
    
    if analysis_type != "Text Summarization":
        st.subheader("NER Settings")
        entity_types = st.multiselect(
            "Select entity types to display:",
            ["PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "MONEY", "PERCENT", "PRODUCT", "EVENT"],
            default=["PERSON", "ORG", "GPE", "DATE"]
        )
    
    if analysis_type != "Named Entity Recognition":
        st.subheader("Summarization Settings")
        summary_ratio = st.slider("Summary length (sentences):", 3, 15, 5)
        summarizer_type = st.selectbox(
            "Choose summarization algorithm:",
            ["LSA", "LexRank", "Luhn", "TextRank"]
        )
    
    st.subheader("Additional Options")
    show_wordcloud = st.checkbox("Show Word Cloud", value=True)
    show_sentiment = st.checkbox("Show Sentiment Analysis", value=True)
    
    st.markdown("---")
    st.info("This tool uses spaCy for NER and sumy for text summarization. You can analyze your text or extract content from a URL.")

# Input options
input_method = st.radio("Choose input method:", ["Enter text", "URL"])

text = ""
if input_method == "Enter text":
    text = st.text_area("Paste your text here:", height=200, 
                       placeholder="Enter or paste your text here...")
else:
    url = st.text_input("Enter URL:", placeholder="https://example.com")
    if url:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
            st.success("Text extracted successfully from URL!")
        except:
            st.error("Error extracting text from URL. Please check the URL and try again.")

# Process the text when the button is clicked
if st.button("Analyze Text") and text:
    with st.spinner("Analyzing text..."):
        # Perform NER
        if analysis_type in ["Named Entity Recognition", "Both"]:
            doc = nlp(text)
            
            # Display NER visualization
            st.markdown("---")
            st.markdown('<div class="highlight">', unsafe_allow_html=True)
            st.subheader("🔍 Named Entity Recognition")
            
            # Create two columns for different visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # HTML rendering of entities
                html = displacy.render(doc, style="ent", page=True)
                st.markdown(html, unsafe_allow_html=True)
            
            with col2:
                # Entity frequency chart
                entities = [ent.text for ent in doc.ents if ent.label_ in entity_types]
                if entities:
                    entity_freq = Counter(entities)
                    top_entities = dict(sorted(entity_freq.items(), key=lambda x: x[1], reverse=True)[:10])
                    
                    fig, ax = plt.subplots()
                    ax.barh(list(top_entities.keys()), list(top_entities.values()))
                    ax.set_xlabel('Frequency')
                    ax.set_title('Top Entity Frequency')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("No entities found with selected types.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Perform text summarization
        if analysis_type in ["Text Summarization", "Both"]:
            st.markdown("---")
            st.markdown('<div class="highlight">', unsafe_allow_html=True)
            st.subheader("📃 Text Summarization")
            
            # Initialize the summarizer based on selection
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            stemmer = Stemmer("english")
            
            if summarizer_type == "LSA":
                summarizer = LsaSummarizer(stemmer)
            elif summarizer_type == "LexRank":
                summarizer = LexRankSummarizer(stemmer)
            elif summarizer_type == "Luhn":
                summarizer = LuhnSummarizer(stemmer)
            else:  # TextRank
                summarizer = TextRankSummarizer(stemmer)
            
            summarizer.stop_words = get_stop_words("english")
            
            # Generate summary
            summary_sentences = summarizer(parser.document, summary_ratio)
            summary = " ".join(str(sentence) for sentence in summary_sentences)
            
            # Display summary
            st.markdown('<div class="summary-box">', unsafe_allow_html=True)
            st.write(summary)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show original and summary statistics
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"Original text: {len(text.split())} words")
            with col2:
                st.success(f"Summary: {len(summary.split())} words")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional visualizations
        if show_wordcloud or show_sentiment:
            st.markdown("---")
            st.subheader("📊 Additional Insights")
            
            if show_wordcloud:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Generate word cloud
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title('Word Cloud')
                    st.pyplot(fig)
                
                with col2:
                    # Generate word frequency
                    words = [token.text for token in nlp(text) if not token.is_stop and token.is_alpha]
                    word_freq = Counter(words)
                    top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10])
                    
                    fig, ax = plt.subplots()
                    ax.barh(list(top_words.keys()), list(top_words.values()))
                    ax.set_xlabel('Frequency')
                    ax.set_title('Top Words (excluding stop words)')
                    plt.tight_layout()
                    st.pyplot(fig)
            
            if show_sentiment:
                # Perform sentiment analysis
                blob = TextBlob(text)
                sentiment = blob.sentiment
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Polarity", f"{sentiment.polarity:.2f}", 
                              help="Polarity: -1 (negative) to +1 (positive)")
                with col2:
                    st.metric("Subjectivity", f"{sentiment.subjectivity:.2f}", 
                              help="Subjectivity: 0 (objective) to 1 (subjective)")
                with col3:
                    if sentiment.polarity > 0.1:
                        sentiment_label = "Positive"
                    elif sentiment.polarity < -0.1:
                        sentiment_label = "Negative"
                    else:
                        sentiment_label = "Neutral"
                    st.metric("Overall Sentiment", sentiment_label)

# Footer
st.markdown("---")
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("### TextInsight - NLP Text Analysis Tool")
st.markdown("Created with ❤️ using Streamlit, spaCy, and sumy")
st.markdown('</div>', unsafe_allow_html=True)