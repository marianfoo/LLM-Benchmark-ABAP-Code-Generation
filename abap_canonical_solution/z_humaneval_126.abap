CLASS z_humaneval_126 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: is_sorted
      IMPORTING
        lst           TYPE int4_table
      RETURNING
        VALUE(result) TYPE abap_bool.

ENDCLASS.

CLASS z_humaneval_126 IMPLEMENTATION.

  METHOD is_sorted.
    DATA: lt_count TYPE SORTED TABLE OF i WITH UNIQUE KEY table_line,
          lv_count TYPE i,
          lv_prev  TYPE i,
          lv_first TYPE abap_bool VALUE abap_true.

    LOOP AT lst INTO DATA(lv_value).
      READ TABLE lt_count TRANSPORTING NO FIELDS WITH KEY table_line = lv_value.
      IF sy-subrc = 0.
        lv_count = lv_count + 1.
        IF lv_count > 1.
          result = abap_false.
          RETURN.
        ENDIF.
      ELSE.
        INSERT lv_value INTO TABLE lt_count.
        lv_count = 0.
      ENDIF.

      IF lv_first = abap_true.
        lv_first = abap_false.
      ELSE.
        IF lv_value < lv_prev.
          result = abap_false.
          RETURN.
        ENDIF.
      ENDIF.
      lv_prev = lv_value.
    ENDLOOP.

    result = abap_true.
  ENDMETHOD.

ENDCLASS.