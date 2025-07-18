CLASS z_humaneval_143 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: words_in_sentence
      IMPORTING
        sentence      TYPE string
      RETURNING
        VALUE(result) TYPE string.
ENDCLASS.

CLASS z_humaneval_143 IMPLEMENTATION.
  METHOD words_in_sentence.
    DATA: word_table TYPE string_table,
          new_lst    TYPE string_table,
          word       TYPE string,
          flg        TYPE i,
          i          TYPE i,
          word_len   TYPE i.

    SPLIT sentence AT ' ' INTO TABLE word_table.

    LOOP AT word_table INTO word.
      flg = 0.
      word_len = strlen( word ).

      IF word_len = 1.
        flg = 1.
      ENDIF.

      DO word_len - 2 TIMES.
        i = sy-index + 1.
        IF word_len MOD i = 0.
          flg = 1.
          EXIT.
        ENDIF.
      ENDDO.

      IF flg = 0 OR word_len = 2.
        APPEND word TO new_lst.
      ENDIF.
    ENDLOOP.

    LOOP AT new_lst INTO word.
      IF sy-tabix = 1.
        result = word.
      ELSE.
        CONCATENATE result word INTO result SEPARATED BY ' '.
      ENDIF.
    ENDLOOP.
  ENDMETHOD.
ENDCLASS.