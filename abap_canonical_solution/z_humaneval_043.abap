CLASS z_humaneval_043 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: pairs_sum_to_zero
      IMPORTING
        VALUE(l)      TYPE int4_table
      RETURNING
        VALUE(result) TYPE abap_bool.
ENDCLASS.

CLASS z_humaneval_043 IMPLEMENTATION.
  METHOD pairs_sum_to_zero.
    DATA: lv_i    TYPE i,
          lv_j    TYPE i,
          lv_size TYPE i.

    lv_size = lines( l ).

    DO lv_size TIMES.
      lv_i = sy-index.
      lv_j = lv_i + 1.
      WHILE lv_j <= lv_size.
        IF l[ lv_i ] + l[ lv_j ] = 0.
          result = abap_true.
          RETURN.
        ENDIF.
        lv_j = lv_j + 1.
      ENDWHILE.
    ENDDO.

    result = abap_false.
  ENDMETHOD.
ENDCLASS.