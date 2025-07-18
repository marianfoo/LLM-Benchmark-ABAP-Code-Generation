CLASS ltcl_test_below_threshold DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_below_threshold IMPLEMENTATION.

  METHOD test_case_1.
    DATA: l TYPE TABLE OF i.
    l = VALUE #( ( 1 ) ( 2 ) ( 4 ) ( 10 ) ).
    cl_abap_unit_assert=>assert_true( z_humaneval_052=>below_threshold( l = l t = 100 ) ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: l TYPE TABLE OF i.
    l = VALUE #( ( 1 ) ( 20 ) ( 4 ) ( 10 ) ).
    cl_abap_unit_assert=>assert_false( z_humaneval_052=>below_threshold( l = l t = 5 ) ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: l TYPE TABLE OF i.
    l = VALUE #( ( 1 ) ( 20 ) ( 4 ) ( 10 ) ).
    cl_abap_unit_assert=>assert_true( z_humaneval_052=>below_threshold( l = l t = 21 ) ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: l TYPE TABLE OF i.
    l = VALUE #( ( 1 ) ( 20 ) ( 4 ) ( 10 ) ).
    cl_abap_unit_assert=>assert_true( z_humaneval_052=>below_threshold( l = l t = 22 ) ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: l TYPE TABLE OF i.
    l = VALUE #( ( 1 ) ( 8 ) ( 4 ) ( 10 ) ).
    cl_abap_unit_assert=>assert_true( z_humaneval_052=>below_threshold( l = l t = 11 ) ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: l TYPE TABLE OF i.
    l = VALUE #( ( 1 ) ( 8 ) ( 4 ) ( 10 ) ).
    cl_abap_unit_assert=>assert_false( z_humaneval_052=>below_threshold( l = l t = 10 ) ).
  ENDMETHOD.

ENDCLASS.