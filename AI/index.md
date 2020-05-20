---
layout: page
title: AI
background: '/img/bg-post.jpg'
pagination: 
  enabled: true
  category: ai
---
<h1>AI</h1>

{% for post in paginator.posts %}

<article class="post-preview">
  <a href="{{ post.url | prepend: site.baseurl | replace: '//', '/' }}">
    <h2 class="post-title">{{ post.title }}</h2>
    {% if post.subtitle %}
    <h3 class="post-subtitle">{{ post.subtitle }}</h3>
    {% else %}
    <h3 class="post-subtitle">{{ post.excerpt | strip_html | truncatewords: 15 }}</h3>
    {% endif %}
  </a>
  <p class="post-meta">Posteado por
    {% if post.author %}
    {{ post.author }}
    {% else %}
    {{ site.author }}
    {% endif %}
    el {{ post.date | date: '%B %d, %Y' }} &middot; {% include read_time.html content=post.content %}
  </p>
</article>

<hr>

{% endfor %}

<!-- Pager -->
{% if paginator.total_pages > 1 %}

<div class="clearfix">

  {% if paginator.previous_page %}
  <a class="btn btn-primary float-left" href="{{ paginator.previous_page_path | prepend: site.baseurl | replace: '//', '/' }}">&larr;
    Nuevos<span class="d-none d-md-inline"> Posts</span></a>
  {% endif %}

  {% if paginator.next_page %}
  <a class="btn btn-primary float-right" href="{{ paginator.next_page_path | prepend: site.baseurl | replace: '//', '/' }}">Antiguos<span class="d-none d-md-inline"> Posts</span> &rarr;</a>
  {% endif %}

</div>

{% endif %}