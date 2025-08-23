CLASS z_humaneval_111 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    TYPES: BEGIN OF ty_result,
             letter TYPE string,
             count  TYPE i,
           END OF ty_result.
    TYPES: tt_result TYPE TABLE OF ty_result WITH EMPTY KEY.

    CLASS-METHODS: histogram
      IMPORTING
        test          TYPE string
      RETURNING
        VALUE(et_result) TYPE tt_result.

  PRIVATE SECTION.

ENDCLASS.

CLASS z_humaneval_111 IMPLEMENTATION.

  METHOD histogram.
    DATA: lt_letters   TYPE TABLE OF string,
          lv_letter    TYPE string,
          lv_count     TYPE i,
          lv_max_count TYPE i,
          ls_result    TYPE ty_result.

    CLEAR et_result.

    IF test IS INITIAL.
      RETURN.
    ENDIF.

    SPLIT test AT ' ' INTO TABLE lt_letters.

    LOOP AT lt_letters INTO lv_letter.
      IF lv_letter IS INITIAL.
        CONTINUE.
      ENDIF.
      lv_count = 0.
      LOOP AT lt_letters INTO DATA(lv_temp).
        IF lv_temp = lv_letter.
          lv_count = lv_count + 1.
        ENDIF.
      ENDLOOP.
      IF lv_count > lv_max_count.
        lv_max_count = lv_count.
      ENDIF.
    ENDLOOP.

    IF lv_max_count > 0.
      LOOP AT lt_letters INTO lv_letter.
        IF lv_letter IS INITIAL.
          CONTINUE.
        ENDIF.
        lv_count = 0.
        LOOP AT lt_letters INTO lv_temp.
          IF lv_temp = lv_letter.
            lv_count = lv_count + 1.
          ENDIF.
        ENDLOOP.
        IF lv_count = lv_max_count.
          READ TABLE et_result INTO ls_result WITH KEY letter = lv_letter.
          IF sy-subrc <> 0.
            ls_result-letter = lv_letter.
            ls_result-count = lv_count.
            APPEND ls_result TO et_result.
          ENDIF.
        ENDIF.
      ENDLOOP.
    ENDIF.

  ENDMETHOD.

ENDCLASS.