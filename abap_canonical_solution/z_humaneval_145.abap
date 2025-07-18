CLASS z_humaneval_145 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS order_by_points
      IMPORTING
        nums          TYPE int4_table
      RETURNING
        VALUE(result) TYPE int4_table.

  PRIVATE SECTION.
    CLASS-METHODS digits_sum
      IMPORTING
        n          TYPE i
      RETURNING
        VALUE(sum) TYPE i.

ENDCLASS.

CLASS z_humaneval_145 IMPLEMENTATION.

  METHOD order_by_points.
    DATA: lt_temp   TYPE TABLE OF REF TO cl_abap_typedescr,
          lt_sorted TYPE TABLE OF i,
          lv_index  TYPE i,
          lv_sum    TYPE i.

    DATA: BEGIN OF ls_item,
            value TYPE i,
            sum   TYPE i,
            index TYPE i,
          END OF ls_item,
          lt_items LIKE TABLE OF ls_item.

    LOOP AT nums INTO DATA(lv_num).
      lv_index = sy-tabix.
      lv_sum = digits_sum( lv_num ).
      ls_item-value = lv_num.
      ls_item-sum = lv_sum.
      ls_item-index = lv_index.
      APPEND ls_item TO lt_items.
    ENDLOOP.

    SORT lt_items BY sum ASCENDING index ASCENDING.

    LOOP AT lt_items INTO ls_item.
      result = VALUE #( BASE result ( ls_item-value ) ).
    ENDLOOP.

  ENDMETHOD.

  METHOD digits_sum.
    DATA: lv_n     TYPE i,
          lv_neg   TYPE i VALUE 1,
          lv_digit TYPE i,
          lv_str   TYPE string,
          lv_len   TYPE i,
          lv_pos   TYPE i.

    lv_n = n.
    IF lv_n < 0.
      lv_n = lv_n * -1.
      lv_neg = -1.
    ENDIF.

    lv_str = |{ lv_n }|.
    lv_len = strlen( lv_str ).

    DO lv_len TIMES.
      lv_pos = sy-index - 1.
      lv_digit = lv_str+lv_pos(1).
      IF sy-index = 1.
        lv_digit = lv_digit * lv_neg.
      ENDIF.
      sum = sum + lv_digit.
    ENDDO.

  ENDMETHOD.

ENDCLASS.