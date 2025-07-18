CLASS z_humaneval_136 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    TYPES: BEGIN OF ty_result,
             largest_negative       TYPE i,
             smallest_positive      TYPE i,
             largest_negative_null  TYPE abap_bool,
             smallest_positive_null TYPE abap_bool,
           END OF ty_result.

    CLASS-METHODS: largest_smallest_integers
      IMPORTING
        iv_list          TYPE int4_table
      RETURNING
        VALUE(rs_result) TYPE ty_result.

  PROTECTED SECTION.
  PRIVATE SECTION.

ENDCLASS.

CLASS z_humaneval_136 IMPLEMENTATION.

  METHOD largest_smallest_integers.
    DATA: lt_negative TYPE int4_table,
          lt_positive TYPE int4_table.

    rs_result-largest_negative_null = abap_true.
    rs_result-smallest_positive_null = abap_true.

    LOOP AT iv_list INTO DATA(lv_value).
      IF lv_value < 0.
        lt_negative = VALUE #( BASE lt_negative ( lv_value ) ).
      ELSEIF lv_value > 0.
        lt_positive = VALUE #( BASE lt_positive ( lv_value ) ).
      ENDIF.
    ENDLOOP.

    IF lines( lt_negative ) > 0.
      SORT lt_negative DESCENDING.
      rs_result-largest_negative = lt_negative[ 1 ].
      rs_result-largest_negative_null = abap_false.
    ENDIF.

    IF lines( lt_positive ) > 0.
      SORT lt_positive ASCENDING.
      rs_result-smallest_positive = lt_positive[ 1 ].
      rs_result-smallest_positive_null = abap_false.
    ENDIF.

  ENDMETHOD.

ENDCLASS.