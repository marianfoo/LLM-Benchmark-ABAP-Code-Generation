CLASS ltcl_test_z_humaneval_032 DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    TYPES: BEGIN OF ty_float_table,
             value TYPE f,
           END OF ty_float_table.
    TYPES: tt_float_table TYPE TABLE OF ty_float_table.

    DATA: coeffs TYPE tt_float_table.

    METHODS: setup,
      poly
        IMPORTING
          xs            TYPE tt_float_table
          x             TYPE f
        RETURNING
          VALUE(result) TYPE f,
      test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING,
      test_case_8 FOR TESTING,
      test_case_9 FOR TESTING,
      test_case_10 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_z_humaneval_032 IMPLEMENTATION.

  METHOD setup.
    CLEAR coeffs.
  ENDMETHOD.

  METHOD poly.
    DATA: sum   TYPE f,
          power TYPE f,
          i     TYPE i.

    sum = 0.
    i = 0.

    LOOP AT xs INTO DATA(coeff).
      power = x ** i.
      sum = sum + coeff-value * power.
      i = i + 1.
    ENDLOOP.

    result = sum.
  ENDMETHOD.

  METHOD test_case_1.
    coeffs = VALUE #( ( value = 1 ) ( value = 2 ) ).
    DATA(solution) = z_humaneval_032=>find_zero( coeffs ).
    DATA(poly_result) = poly( xs = coeffs x = solution ).
    cl_abap_unit_assert=>assert_true( act = COND #( WHEN abs( poly_result ) < + '1E-4' THEN abap_true ELSE abap_false ) ).
  ENDMETHOD.

  METHOD test_case_2.
    coeffs = VALUE #( ( value = -6 ) ( value = 11 ) ( value = -6 ) ( value = 1 ) ).
    DATA(solution) = z_humaneval_032=>find_zero( coeffs ).
    DATA(poly_result) = poly( xs = coeffs x = solution ).
    cl_abap_unit_assert=>assert_true( act = COND #( WHEN abs( poly_result ) < + '1E-4' THEN abap_true ELSE abap_false ) ).
  ENDMETHOD.

  METHOD test_case_3.
    coeffs = VALUE #( ( value = 3 ) ( value = -2 ) ).
    DATA(solution) = z_humaneval_032=>find_zero( coeffs ).
    DATA(poly_result) = poly( xs = coeffs x = solution ).
    cl_abap_unit_assert=>assert_true( act = COND #( WHEN abs( poly_result ) < + '1E-4' THEN abap_true ELSE abap_false ) ).
  ENDMETHOD.

  METHOD test_case_4.
    coeffs = VALUE #( ( value = -8 ) ( value = 5 ) ( value = 1 ) ( value = -2 ) ).
    DATA(solution) = z_humaneval_032=>find_zero( coeffs ).
    DATA(poly_result) = poly( xs = coeffs x = solution ).
    cl_abap_unit_assert=>assert_true( act = COND #( WHEN abs( poly_result ) < + '1E-4' THEN abap_true ELSE abap_false ) ).
  ENDMETHOD.

  METHOD test_case_5.
    coeffs = VALUE #( ( value = 7 ) ( value = -3 ) ( value = 4 ) ( value = -1 ) ).
    DATA(solution) = z_humaneval_032=>find_zero( coeffs ).
    DATA(poly_result) = poly( xs = coeffs x = solution ).
    cl_abap_unit_assert=>assert_true( act = COND #( WHEN abs( poly_result ) < + '1E-4' THEN abap_true ELSE abap_false ) ).
  ENDMETHOD.

  METHOD test_case_6.
    coeffs = VALUE #( ( value = -5 ) ( value = 9 ) ( value = -7 ) ( value = 2 ) ( value = 1 ) ( value = -4 ) ).
    DATA(solution) = z_humaneval_032=>find_zero( coeffs ).
    DATA(poly_result) = poly( xs = coeffs x = solution ).
    cl_abap_unit_assert=>assert_true( act = COND #( WHEN abs( poly_result ) < + '1E-4' THEN abap_true ELSE abap_false ) ).
  ENDMETHOD.

  METHOD test_case_7.
    coeffs = VALUE #( ( value = 4 ) ( value = -8 ) ( value = 6 ) ( value = -2 ) ).
    DATA(solution) = z_humaneval_032=>find_zero( coeffs ).
    DATA(poly_result) = poly( xs = coeffs x = solution ).
    cl_abap_unit_assert=>assert_true( act = COND #( WHEN abs( poly_result ) < + '1E-4' THEN abap_true ELSE abap_false ) ).
  ENDMETHOD.

  METHOD test_case_8.
    coeffs = VALUE #( ( value = -10 ) ( value = 3 ) ( value = 8 ) ( value = -1 ) ( value = 2 ) ( value = -6 ) ).
    DATA(solution) = z_humaneval_032=>find_zero( coeffs ).
    DATA(poly_result) = poly( xs = coeffs x = solution ).
    cl_abap_unit_assert=>assert_true( act = COND #( WHEN abs( poly_result ) < + '1E-4' THEN abap_true ELSE abap_false ) ).
  ENDMETHOD.

  METHOD test_case_9.
    coeffs = VALUE #( ( value = 2 ) ( value = -5 ) ( value = 3 ) ( value = -1 ) ).
    DATA(solution) = z_humaneval_032=>find_zero( coeffs ).
    DATA(poly_result) = poly( xs = coeffs x = solution ).
    cl_abap_unit_assert=>assert_true( act = COND #( WHEN abs( poly_result ) < + '1E-4' THEN abap_true ELSE abap_false ) ).
  ENDMETHOD.

  METHOD test_case_10.
    coeffs = VALUE #( ( value = -1 ) ( value = 7 ) ( value = -4 ) ( value = 9 ) ( value = -2 ) ( value = 1 ) ).
    DATA(solution) = z_humaneval_032=>find_zero( coeffs ).
    DATA(poly_result) = poly( xs = coeffs x = solution ).
    cl_abap_unit_assert=>assert_true( act = COND #( WHEN abs( poly_result ) < + '1E-4' THEN abap_true ELSE abap_false ) ).
  ENDMETHOD.

ENDCLASS.