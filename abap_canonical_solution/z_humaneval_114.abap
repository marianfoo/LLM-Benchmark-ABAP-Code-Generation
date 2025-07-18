CLASS z_humaneval_114 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: min_sub_array_sum
      IMPORTING
        nums          TYPE int4_table
      RETURNING
        VALUE(result) TYPE i.

ENDCLASS.

CLASS z_humaneval_114 IMPLEMENTATION.

  METHOD min_sub_array_sum.
    DATA: max_sum TYPE i VALUE 0,
          s       TYPE i VALUE 0,
          num     TYPE i,
          max_neg TYPE i.

    LOOP AT nums INTO num.
      s = s + ( -1 * num ).
      IF s < 0.
        s = 0.
      ENDIF.
      IF s > max_sum.
        max_sum = s.
      ENDIF.
    ENDLOOP.

    IF max_sum = 0.
      LOOP AT nums INTO num.
        IF sy-tabix = 1.
          max_neg = -1 * num.
        ELSE.
          IF ( -1 * num ) > max_neg.
            max_neg = -1 * num.
          ENDIF.
        ENDIF.
      ENDLOOP.
      max_sum = max_neg.
    ENDIF.

    result = -1 * max_sum.

  ENDMETHOD.

ENDCLASS.