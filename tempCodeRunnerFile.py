    if preds != 'ham':
            return render_template('index.html', value="Spam")
        else:
            re